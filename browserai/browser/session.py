"""Async Playwright browser session.

Owns the browser, context and tabs, and turns browser-level happenings the agent can't
see in the DOM (JS dialogs, new/closed tabs, blocked downloads) into short event messages
that the agent loop reports to the model.
"""

from __future__ import annotations

import asyncio
from types import TracebackType

from playwright.async_api import (
    Browser,
    BrowserContext,
    Dialog,
    Download,
    Page,
    Playwright,
    Route,
    async_playwright,
)

from browserai.agent.safety import domain_allowed
from browserai.browser.waits import DOM_TRACKER_JS, NetworkMonitor, SettleResult, settle
from browserai.config import Settings


class BrowserSession:
    """Usage:
        async with BrowserSession(settings) as session:
            await session.goto("https://example.com")
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._monitors: dict[Page, NetworkMonitor] = {}
        self.events: list[str] = []
        # One-shot answer for the next confirm()/prompt() dialog, set by the agent.
        self.next_dialog: tuple[bool, str | None] | None = None

    async def __aenter__(self) -> BrowserSession:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    # ---------- lifecycle ----------

    async def start(self) -> None:
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=self.settings.headless)
        self._context = await self._browser.new_context(
            viewport={"width": self.settings.viewport_width, "height": self.settings.viewport_height},
            accept_downloads=False,  # the agent never saves files to disk
        )
        self._context.set_default_navigation_timeout(self.settings.nav_timeout_ms)
        await self._context.add_init_script(DOM_TRACKER_JS)
        if self.settings.allowed_domains:
            # Enforce the allowlist at the network layer: this covers typed URLs, clicked
            # links, redirects and pop-ups alike. (Routing disables Playwright's HTTP cache,
            # so it is only switched on when an allowlist is configured.)
            await self._context.route("**/*", self._domain_filter)
        self._context.on("page", self._on_new_page)
        page = await self._context.new_page()
        self.events.clear()  # the first tab is not news
        self._page = page

    async def close(self) -> None:
        # Close in reverse order; tolerate partial startup.
        if self._context is not None:
            await self._context.close()
        if self._browser is not None:
            await self._browser.close()
        if self._pw is not None:
            await self._pw.stop()
        self._pw = self._browser = self._context = self._page = None
        self._monitors.clear()

    # ---------- tabs ----------

    def _on_new_page(self, page: Page) -> None:
        self._monitors[page] = NetworkMonitor(page)
        page.on("dialog", self._on_dialog)
        page.on("download", self._on_download)
        page.on("close", self._on_close)
        # New tabs (target=_blank, window.open) become the active page automatically.
        self._page = page
        self.events.append("a new tab opened and is now the active tab")

    def _on_close(self, page: Page) -> None:
        self._monitors.pop(page, None)
        if page is self._page:
            remaining = self.pages
            self._page = remaining[-1] if remaining else None
            self.events.append("the active tab was closed; switched to the previous tab")

    @property
    def pages(self) -> list[Page]:
        return [p for p in (self._context.pages if self._context else []) if not p.is_closed()]

    @property
    def page(self) -> Page:
        if self._page is None or self._page.is_closed():
            remaining = self.pages
            if not remaining:
                raise RuntimeError("BrowserSession has no open tabs")
            self._page = remaining[-1]
        return self._page

    async def switch_tab(self, index: int) -> Page:
        pages = self.pages
        if not 0 <= index < len(pages):
            raise IndexError(f"there is no tab {index}; open tabs are 0-{len(pages) - 1}")
        self._page = pages[index]
        await self._page.bring_to_front()
        return self._page

    async def close_tab(self) -> None:
        if len(self.pages) <= 1:
            raise RuntimeError("cannot close the last tab")
        await self.page.close()

    async def tabs_text(self) -> str | None:
        """One line per tab, or None when only one tab is open."""
        pages = self.pages
        if len(pages) <= 1:
            return None
        lines = ["Open tabs:"]
        for i, p in enumerate(pages):
            try:
                title = await p.title()
            except Exception:
                title = ""
            mark = " (active)" if p is self._page else ""
            lines.append(f"  tab {i}: {title[:60]!r} {p.url[:80]}{mark}")
        return "\n".join(lines)

    # ---------- dialogs / downloads ----------

    async def _on_dialog(self, dialog: Dialog) -> None:
        kind, message = dialog.type, dialog.message
        if self.next_dialog is not None:
            accept, text = self.next_dialog
            self.next_dialog = None
        else:
            # Alerts only need acknowledging. confirm()/prompt() often guard destructive
            # actions, so they are cancelled unless the agent explicitly asked to accept.
            accept, text = kind in ("alert", "beforeunload"), None
        if accept:
            if kind == "prompt" and text is not None:
                await dialog.accept(text)
            else:
                await dialog.accept()
        else:
            await dialog.dismiss()

        outcome = "accepted" if accept else "dismissed (Cancel)"
        msg = f'{kind} dialog said "{message[:200]}" - {outcome}'
        if not accept:
            msg += ". To accept it instead, call handle_next_dialog(accept=true) and repeat the action"
        self.events.append(msg)

    def _on_download(self, download: Download) -> None:
        self.events.append(f'a download of "{download.suggested_filename}" was blocked (downloads are disabled)')

    async def _domain_filter(self, route: Route) -> None:
        request = route.request
        # Only page navigations are policed; images/scripts/APIs from other hosts (CDNs) are fine.
        if request.is_navigation_request() and not domain_allowed(request.url, self.settings.allowed_domains):
            self.events.append(
                f"navigation to {request.url[:100]} was blocked: outside the allowed domains "
                f"({', '.join(self.settings.allowed_domains)})"
            )
            # A 204 response makes the browser stay on the current page (an abort would leave
            # the tab on Chrome's error page). A pop-up opened straight to a blocked URL stays
            # blank, so it is closed.
            await route.fulfill(status=204, body="")
            page = request.frame.page
            if page.url == "about:blank" and len(self.pages) > 1:
                asyncio.get_running_loop().create_task(page.close())
            return
        await route.continue_()

    def drain_events(self) -> list[str]:
        events, self.events = self.events, []
        return events

    # ---------- navigation helpers ----------

    async def goto(self, url: str) -> None:
        if "://" not in url:
            url = "https://" + url
        await self.page.goto(url, wait_until="domcontentloaded")

    async def settle(self, **kwargs) -> SettleResult:
        page = self.page
        return await settle(page, self._monitors.get(page), timeout=self.settings.settle_timeout_s, **kwargs)

    async def title(self) -> str:
        return await self.page.title()

    async def visible_text(self, limit: int = 4000) -> str:
        text = await self.page.evaluate("() => document.body ? document.body.innerText : ''")
        return " ".join(text.split())[:limit]

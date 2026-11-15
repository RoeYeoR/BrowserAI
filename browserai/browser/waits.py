"""Deciding when a page has "settled" after an action.

Single-page apps rarely fire a useful load event: a click starts fetches, the framework
re-renders, maybe more fetches follow. We consider the page settled when BOTH:

  * network is quiet - no relevant request in flight for `net_quiet` seconds
    (websockets, event streams, media, beacons and long-running requests such as
    long-polling are ignored so they can't block forever), and
  * the DOM is quiet - no structural/text mutation for `dom_quiet` seconds
    (style-only mutations from animations are ignored),

or when `timeout` expires, whichever comes first.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page, Request

# Installed in every document (via add_init_script) and lazily into the current one.
DOM_TRACKER_JS = """
(() => {
  if (window.__baiMutInstalled) return;
  window.__baiMutInstalled = true;
  window.__baiLastMut = performance.now();
  const IGNORED_ATTRS = new Set(['style', 'data-bai-id']);
  new MutationObserver((records) => {
    for (const r of records) {
      if (r.type === 'attributes' && IGNORED_ATTRS.has(r.attributeName)) continue;
      window.__baiLastMut = performance.now();
      return;
    }
  }).observe(document, { subtree: true, childList: true, attributes: true, characterData: true });
})()
"""

_DOM_IDLE_MS_JS = (
    "() => { if (!window.__baiMutInstalled) return -1;"
    " return performance.now() - window.__baiLastMut; }"
)

_IGNORED_RESOURCE_TYPES = {"websocket", "eventsource", "media", "ping", "manifest", "other"}


class NetworkMonitor:
    """Tracks in-flight requests for one page."""

    def __init__(self, page: Page, long_request_s: float = 2.5) -> None:
        self.long_request_s = long_request_s
        self._inflight: dict[Request, float] = {}
        self._last_activity = time.monotonic()
        page.on("request", self._on_start)
        page.on("requestfinished", self._on_end)
        page.on("requestfailed", self._on_end)

    def _on_start(self, req: Request) -> None:
        if req.resource_type in _IGNORED_RESOURCE_TYPES:
            return
        self._inflight[req] = time.monotonic()
        self._last_activity = time.monotonic()

    def _on_end(self, req: Request) -> None:
        if self._inflight.pop(req, None) is not None:
            self._last_activity = time.monotonic()

    def pending(self) -> int:
        """Requests that still count as "loading" (young enough not to be long-polls)."""
        now = time.monotonic()
        return sum(1 for started in self._inflight.values() if now - started < self.long_request_s)

    def quiet_for(self) -> float:
        if self.pending():
            return 0.0
        return time.monotonic() - self._last_activity


@dataclass
class SettleResult:
    seconds: float
    timed_out: bool
    pending_requests: int = 0


async def settle(
    page: Page,
    monitor: NetworkMonitor | None = None,
    *,
    timeout: float = 6.0,
    net_quiet: float = 0.4,
    dom_quiet: float = 0.3,
) -> SettleResult:
    start = time.monotonic()
    deadline = start + timeout
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=timeout * 1000)
    except PlaywrightError:
        pass

    while True:
        now = time.monotonic()
        if now >= deadline:
            return SettleResult(now - start, True, monitor.pending() if monitor else 0)

        net_ok = monitor is None or monitor.quiet_for() >= net_quiet
        try:
            idle_ms = await page.evaluate(_DOM_IDLE_MS_JS)
            if idle_ms < 0:  # tracker missing (document loaded before we attached): install now
                await page.evaluate(DOM_TRACKER_JS)
                idle_ms = 0
        except PlaywrightError:
            idle_ms = 0  # context destroyed by a navigation; the new document will report soon
        if net_ok and idle_ms >= dom_quiet * 1000:
            return SettleResult(time.monotonic() - start, False)
        await asyncio.sleep(0.1)

"""SPA robustness: settle waits, re-finding re-rendered elements, dialogs, tabs and downloads.
Uses a local HTTP server (see conftest.server) with an artificially slow API endpoint."""

import time

import pytest

from browserai.agent.safety import SafetyGuard
from browserai.agent.tools import ToolExecutor
from browserai.browser import BrowserSession
from browserai.config import Settings
from browserai.perception import DomExtractor


@pytest.fixture
async def shop(server):
    async with BrowserSession(Settings(headless=True)) as session:
        ex = ToolExecutor(session, SafetyGuard(mode="allow"))  # flows the user approved
        ex.extractor = DomExtractor()

        async def snap():
            ex.state = await ex.extractor.snapshot(session.page)
            return ex.state

        async def go(path: str):
            await session.page.goto(server + path, wait_until="load")
            await session.settle()
            return await snap()

        ex.snap, ex.go = snap, go
        yield ex


def idx(state, name):
    return next(e.index for e in state.elements if e.name == name)


def names(state):
    return {e.name for e in state.elements}


async def status(ex):
    return await ex.page.inner_text("#status")


# ---------- settle ----------

async def test_settle_waits_for_api_and_render(shop, server):
    await shop.session.page.goto(server + "/spa.html#/products", wait_until="load")
    early = await shop.snap()
    assert "Add Widget" not in names(early)  # still "Loading…": a plain load-event wait is not enough

    result = await shop.session.settle()
    assert not result.timed_out
    assert result.seconds >= 0.3
    assert "Add Widget" in names(await shop.snap())


async def test_settle_ignores_long_running_request(shop):
    t0 = time.monotonic()
    await shop.go("/longpoll.html")  # page keeps a 10 s request open
    assert time.monotonic() - t0 < 5


# ---------- re-finding re-rendered elements ----------

async def test_click_rerendered_element_by_old_number(shop):
    state = await shop.go("/spa.html#/products")
    widget = idx(state, "Add Widget")

    r1 = await shop.execute("click", {"index": widget})
    assert not r1.is_error and "re-found" not in r1.text
    # The click re-rendered the list, so the node behind [widget] is gone; acting on the OLD
    # number without a new snapshot must still work.
    r2 = await shop.execute("click", {"index": widget})
    assert not r2.is_error, r2.text
    assert "re-found" in r2.text
    assert await shop.page.inner_text("#cart") == "Cart: 2"

    # ...and the element keeps its number in the next snapshot.
    assert idx(await shop.snap(), "Add Widget") == widget


async def test_refind_never_steals_another_element(shop):
    state = await shop.go("/spa.html#/products")
    gadget = idx(state, "Add Gadget")
    await shop.page.evaluate("() => document.querySelectorAll('#app .p')[1].remove()")  # Gadget gone
    r = await shop.execute("click", {"index": gadget})
    assert r.is_error and "no longer on the page" in r.text
    assert await shop.page.inner_text("#cart") == "Cart: 0"


async def test_snapshot_keeps_numbers_across_rerender(shop):
    state = await shop.go("/spa.html#/products")
    before = {n: idx(state, n) for n in ("Add Widget", "Add Gadget", "Add Doohickey")}
    await shop.page.evaluate("() => renderProducts()")  # every product node replaced
    after = await shop.snap()
    assert {n: idx(after, n) for n in before} == before


# ---------- dialogs ----------

async def test_alert_is_accepted_and_reported(shop):
    state = await shop.go("/spa.html")
    await shop.execute("click", {"index": idx(state, "Show alert")})
    assert await status(shop) == "alerted"
    events = shop.session.drain_events()
    assert any('alert dialog said "Saved!" - accepted' in e for e in events)


async def test_confirm_cancelled_by_default_then_accepted_on_request(shop):
    state = await shop.go("/spa.html")
    delete = idx(state, "Delete all")

    await shop.execute("click", {"index": delete})
    assert await status(shop) == "kept"
    events = shop.session.drain_events()
    assert any("confirm dialog" in e and "dismissed" in e and "handle_next_dialog" in e for e in events)

    r = await shop.execute("handle_next_dialog", {"accept": True})
    assert not r.is_error
    await shop.execute("click", {"index": delete})
    assert await status(shop) == "deleted"
    # The one-shot answer is used up: the next confirm is cancelled again.
    await shop.execute("click", {"index": delete})
    assert await status(shop) == "kept"


async def test_prompt_with_text(shop):
    state = await shop.go("/spa.html")
    await shop.execute("handle_next_dialog", {"accept": True, "prompt_text": "Roi"})
    await shop.execute("click", {"index": idx(state, "Ask name")})
    assert await status(shop) == "hello Roi"


# ---------- tabs and downloads ----------

async def test_new_tab_switch_and_close(shop):
    state = await shop.go("/spa.html")
    first = shop.session.page
    await shop.execute("click", {"index": idx(state, "Open help in new tab")})
    await shop.session.settle()  # the popup opens a moment after the click returns

    assert shop.session.page is not first
    assert shop.session.page.url.endswith("/help.html")
    assert any("new tab" in e for e in shop.session.drain_events())
    tabs = await shop.session.tabs_text()
    assert "tab 0" in tabs and "tab 1" in tabs and "(active)" in tabs

    r = await shop.execute("switch_tab", {"tab": 0})
    assert not r.is_error and shop.session.page is first
    assert (await shop.execute("switch_tab", {"tab": 5})).is_error

    await shop.execute("switch_tab", {"tab": 1})
    r = await shop.execute("close_tab", {})
    assert not r.is_error
    assert shop.session.page is first and len(shop.session.pages) == 1
    assert (await shop.execute("close_tab", {})).is_error  # never close the last tab


async def test_page_closing_itself_returns_to_previous_tab(shop):
    state = await shop.go("/spa.html")
    first = shop.session.page
    await shop.execute("click", {"index": idx(state, "Open help in new tab")})
    await shop.session.settle()  # the popup opens a moment after the click returns
    help_state = await shop.snap()
    shop.session.drain_events()

    await shop.execute("click", {"index": idx(help_state, "Close this tab")})
    await shop.session.page.wait_for_timeout(300)
    assert shop.session.page is first
    assert any("closed" in e for e in shop.session.drain_events())


async def test_download_is_blocked_and_reported(shop):
    state = await shop.go("/spa.html")
    await shop.execute("click", {"index": idx(state, "Download file")})
    await shop.page.wait_for_timeout(500)
    assert any('download of "hello.txt" was blocked' in e for e in shop.session.drain_events())

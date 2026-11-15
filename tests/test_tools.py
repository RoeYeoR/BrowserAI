"""Tests for the tool layer: argument validation and real browser actions on a local page."""

import pytest

from browserai.agent.tools import TOOLS, ToolExecutor, tool_definitions
from browserai.browser import BrowserSession
from browserai.config import Settings
from browserai.perception import DomExtractor

from conftest import PAGES


@pytest.fixture
async def ex():
    async with BrowserSession(Settings(headless=True)) as session:
        await session.page.goto((PAGES / "actions.html").as_uri(), wait_until="load")
        executor = ToolExecutor(session)
        executor.state = await DomExtractor().snapshot(session.page)
        yield executor


def idx(executor: ToolExecutor, name: str) -> int:
    return next(e.index for e in executor.state.elements if e.name == name)


async def out(executor: ToolExecutor) -> str:
    return await executor.page.inner_text("#out")


def test_tool_definitions_are_valid():
    defs = tool_definitions()
    assert [d["name"] for d in defs] == list(TOOLS)
    for d in defs:
        assert d["description"], d["name"]
        assert d["input_schema"]["type"] == "object"
        assert "title" not in d["input_schema"]


async def test_click(ex):
    r = await ex.execute("click", {"index": idx(ex, "Press me")})
    assert not r.is_error
    assert await out(ex) == "button clicked"


async def test_type_and_enter(ex):
    r = await ex.execute("type_text", {"index": idx(ex, "Query"), "text": "playwright", "press_enter": True})
    assert not r.is_error, r.text
    assert await out(ex) == "searched: playwright"


async def test_type_append(ex):
    await ex.execute("type_text", {"index": idx(ex, "Query"), "text": "+new", "clear": False})
    assert await ex.page.input_value("#q") == "old+new"


async def test_click_hidden_checkbox_via_label(ex):
    r = await ex.execute("click", {"index": idx(ex, "Fancy checkbox")})
    assert "label" in r.text
    assert await ex.page.is_checked("#fancy")


async def test_disabled_and_unknown_elements_are_errors(ex):
    r = await ex.execute("click", {"index": idx(ex, "Disabled one")})
    assert r.is_error and "disabled" in r.text
    r = await ex.execute("click", {"index": 999})
    assert r.is_error and "no element [999]" in r.text


async def test_removed_element_reports_stale(ex):
    i = idx(ex, "Press me")
    await ex.page.evaluate("() => document.getElementById('btn').remove()")
    r = await ex.execute("click", {"index": i})
    assert r.is_error and "no longer on the page" in r.text


async def test_invalid_arguments(ex):
    r = await ex.execute("click", {"index": "twelve"})
    assert r.is_error and "invalid arguments" in r.text
    r = await ex.execute("scroll", {"direction": "sideways"})
    assert r.is_error
    r = await ex.execute("nope", {})
    assert r.is_error and "unknown tool" in r.text


async def test_scroll_page_and_container(ex):
    r = await ex.execute("scroll", {"direction": "down"})
    assert not r.is_error and "scrolled down" in r.text
    assert await ex.page.evaluate("() => window.scrollY") > 0

    r = await ex.execute("scroll", {"direction": "down", "index": idx(ex, "Inside scroll box")})
    assert "inside the container" in r.text
    assert await ex.page.evaluate("() => document.getElementById('box').scrollTop") > 0


async def test_read_text_and_notes(ex):
    r = await ex.execute("read_text", {})
    assert "nothing yet" in r.text
    await ex.execute("save_note", {"note": "price is $5"})
    assert ex.notes == ["price is $5"]


async def test_done(ex):
    r = await ex.execute("done", {"success": True, "answer": "42"})
    assert r.done is not None and r.done.answer == "42"


async def test_navigate_rejects_non_http(ex):
    r = await ex.execute("navigate", {"url": "file:///etc/passwd"})
    assert r.is_error and "http" in r.text

"""Tests for the visual feedback loop: annotated screenshots, page diffs, and the agent's
verify step (driven by a scripted fake LLM, so no API calls)."""

import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from browserai.agent.loop import NO_CHANGE_WARNING, Agent
from browserai.browser import BrowserSession
from browserai.config import Settings
from browserai.perception import DomExtractor
from browserai.perception.vision import annotate

from conftest import PAGES


def by_name(state, name):
    return next(e for e in state.elements if e.name == name)


# ---------- annotated screenshots ----------

async def test_annotated_screenshot(open_page):
    page = await open_page("basic.html")
    state = await DomExtractor().snapshot(page)
    png = await page.screenshot(type="png")
    jpeg = annotate(png, state)

    img = Image.open(io.BytesIO(jpeg))
    assert img.format == "JPEG"
    assert img.size == (1280, 800)

    # The box outline around the "Go" button is drawn in a strong colour (not the white page).
    go = by_name(state, "Go")
    r, g, b = img.convert("RGB").getpixel((go.box.x + go.box.w // 2, go.box.y + go.box.h))
    assert max(r, g, b) - min(r, g, b) > 60, (r, g, b)


async def test_offscreen_elements_not_drawn(open_page):
    page = await open_page("basic.html")
    state = await DomExtractor().snapshot(page)
    png = await page.screenshot(type="png")
    plain = Image.open(io.BytesIO(png)).convert("RGB")
    # Annotating with only off-screen elements must not draw anything visible.
    state.elements = [e for e in state.elements if not e.in_viewport]
    drawn = Image.open(io.BytesIO(annotate(png, state, quality=95))).convert("RGB")
    diff = sum(abs(a - b) for a, b in zip(plain.tobytes(), drawn.tobytes())) / len(plain.tobytes())
    assert diff < 3  # only JPEG noise


# ---------- diffs ----------

async def test_diff_detects_value_and_text_changes(open_page):
    page = await open_page("actions.html")
    ex = DomExtractor()
    s1 = await ex.snapshot(page)
    assert (await ex.snapshot(page)).diff(s1) == []  # nothing happened

    await page.fill("#q", "hello")
    await page.evaluate("() => document.getElementById('btn').textContent = 'Pressed'")
    changes = " ".join((await ex.snapshot(page)).diff(s1))
    assert 'value: "old" → "hello"' in changes
    assert 'text: "Press me" → "Pressed"' in changes


async def test_diff_new_and_removed_elements_and_scroll(open_page):
    page = await open_page("dynamic.html")
    ex = DomExtractor()
    s1 = await ex.snapshot(page)
    await page.evaluate(
        "() => { document.getElementById('a').remove();"
        " const n = document.createElement('button'); n.textContent = 'C';"
        " document.getElementById('list').append(n); }"
    )
    changes = " ".join((await ex.snapshot(page)).diff(s1))
    assert "1 new element(s)" in changes and '"C"' in changes
    assert "1 element(s) disappeared" in changes and '"A"' in changes


async def test_diff_spa_route_vs_reload(open_page):
    page = await open_page("dynamic.html")
    ex = DomExtractor()
    s1 = await ex.snapshot(page)
    await page.evaluate("() => history.pushState({}, '', '#/next')")
    spa = (await ex.snapshot(page)).diff(s1)
    assert any("URL changed" in c for c in spa)

    s2 = await ex.snapshot(page)
    await page.reload(wait_until="load")
    assert "the page reloaded" in (await ex.snapshot(page)).diff(s2)


# ---------- agent verify step with a fake LLM ----------

def validate_messages(messages):
    """The Messages API rules the agent loop must respect (the real API returns 400 otherwise)."""
    pending = set()
    for m in messages:
        content = m["content"] if isinstance(m["content"], list) else []
        if m["role"] == "assistant":
            pending |= {getattr(b, "id", None) for b in content if getattr(b, "type", None) == "tool_use"}
            continue
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                pending.discard(b["tool_use_id"])
                if b.get("is_error"):
                    kinds = {c["type"] for c in b["content"]}
                    assert kinds <= {"text"}, f"is_error tool_result may only hold text, got {kinds}"
        assert not pending, f"tool_use without tool_result: {pending}"


class FakeLLM:
    """Returns scripted tool calls; records the messages it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def create(self, messages, **kwargs):
        validate_messages(messages)  # enforce the API's structural rules, like the real endpoint
        self.calls.append(kwargs)
        name, args = self.script.pop(0)
        block = SimpleNamespace(type="tool_use", id=f"tu{len(self.calls)}", name=name, input=args)
        return SimpleNamespace(
            content=[block], stop_reason="tool_use", context_management=None,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )


def last_feedback(agent) -> str:
    result_block = agent.messages[-1]["content"][-1]
    return " ".join(c.get("text", "") for c in result_block["content"])


@pytest.fixture
async def session():
    async with BrowserSession(Settings(headless=True)) as s:
        await s.page.goto((PAGES / "actions.html").as_uri(), wait_until="load")
        yield s


async def test_agent_reports_changes_and_no_change(session, tmp_path):
    ex = DomExtractor()
    st = await ex.snapshot(session.page)
    press = by_name(st, "Press me").index
    disabled_noop = by_name(st, "Query").index  # clicking a text field changes nothing visible

    llm = FakeLLM([
        ("click", {"index": press}),
        ("click", {"index": disabled_noop}),
        ("done", {"success": True, "answer": "ok"}),
    ])
    settings = Settings(headless=True, max_steps=5, trace_dir=str(tmp_path))
    agent = Agent("test task", session, llm, settings)

    seen = []
    agent._listeners.append(lambda e: seen.append(e))
    # Capture feedback after each step by wrapping _execute.
    original = agent._execute

    async def spy(tool_uses, step):
        res = await original(tool_uses, step)
        if res[0]:
            seen.append(("feedback", " ".join(c.get("text", "") for c in res[0][-1]["content"])))
        return res

    agent._execute = spy
    result = await agent.run()

    assert result.success and result.answer == "ok" and result.steps == 3
    feedback = [f for kind, f in (x for x in seen if isinstance(x, tuple))]
    assert "Changes since your last action" in feedback[0]
    assert 'new text on the page: "button clicked"' in feedback[0]  # a plain <p>, not an element
    assert NO_CHANGE_WARNING in feedback[1]

    # Every observation carried a screenshot, and the trace folder has the expected files.
    first_msg = agent.messages[0]["content"]
    assert first_msg[-1]["type"] == "image"
    trace = tmp_path / next(p.name for p in tmp_path.iterdir())
    assert {"task.json", "steps.jsonl", "result.json", "step_00.jpg", "step_01.jpg"} <= {p.name for p in trace.iterdir()}
    records = [json.loads(line) for line in (trace / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["step"] for r in records] == [0, 1, 2, 3]
    assert any(e["kind"] == "verify" and e["changed"] is False for e in records[2]["events"])


async def test_failed_last_action_keeps_messages_valid(session):
    """Regression: a failing last action used to get the screenshot inside its is_error result."""
    st = await DomExtractor().snapshot(session.page)
    press = by_name(st, "Press me").index
    llm = FakeLLM([("done", {"success": True, "answer": "x"})])
    # Turn 1: a success followed by a failure. Turn 2: only a failure. Turn 3: done.
    original = llm.create

    async def two_calls(messages, **kwargs):
        if not llm.calls:
            validate_messages(messages)
            llm.calls.append(kwargs)
            blocks = [SimpleNamespace(type="tool_use", id="a", name="click", input={"index": press}),
                      SimpleNamespace(type="tool_use", id="b", name="click", input={"index": 999})]
            return SimpleNamespace(content=blocks, stop_reason="tool_use", context_management=None,
                                   usage=SimpleNamespace(input_tokens=1, output_tokens=1,
                                                         cache_read_input_tokens=0, cache_creation_input_tokens=0))
        if len(llm.calls) == 1:
            validate_messages(messages)
            llm.calls.append(kwargs)
            block = SimpleNamespace(type="tool_use", id="c", name="click", input={"index": 998})  # only a failure
            return SimpleNamespace(content=[block], stop_reason="tool_use", context_management=None,
                                   usage=SimpleNamespace(input_tokens=1, output_tokens=1,
                                                         cache_read_input_tokens=0, cache_creation_input_tokens=0))
        return await original(messages, **kwargs)

    llm.create = two_calls
    agent = Agent("t", session, llm, Settings(headless=True, trace_dir=""))
    result = await agent.run()
    assert result.success  # every request passed validation
    turn1, turn2 = agent.messages[2]["content"], agent.messages[4]["content"]
    assert turn1[1]["is_error"] and turn1[0]["content"][-1]["type"] == "image"  # screenshot on the success
    assert "is_error" not in turn2[0] and turn2[0]["content"][0]["text"].startswith("ERROR: ")


async def test_agent_without_vision(session):
    st = await DomExtractor().snapshot(session.page)
    llm = FakeLLM([("click", {"index": by_name(st, "Press me").index}), ("done", {"success": True, "answer": "x"})])
    agent = Agent("t", session, llm, Settings(headless=True, vision=False, trace_dir=""))
    result = await agent.run()
    assert result.success and result.trace_dir is None
    assert all(c["type"] == "text" for c in agent.messages[0]["content"])

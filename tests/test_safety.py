"""Safety layer: hard blocks on sensitive fields, human approval for risky actions, the domain
allowlist (enforced at the network layer), and the run's step/cost limits."""

from types import SimpleNamespace

import pytest

from browserai.agent.loop import Agent, Usage
from browserai.agent.safety import SafetyGuard, domain_allowed, review
from browserai.agent.tools import ToolExecutor
from browserai.browser import BrowserSession
from browserai.config import Settings
from browserai.perception import DomExtractor
from browserai.perception.state import Box, Element


# ---------- pure rules ----------

def el(name="", kind="button", sensitive=None, index=1):
    return Element(index=index, bai_id=1, kind=kind, tag="button", name=name, box=Box(0, 0, 10, 10),
                   in_viewport=True, sensitive=sensitive)


def test_review_rules():
    assert review("click", {"index": 1}, el("Place order")).needs_approval
    assert review("click", {"index": 1}, el("Delete account")).needs_approval
    assert review("click", {"index": 1}, el("Submit application")).needs_approval
    assert not review("click", {"index": 1}, el("Search")).needs_approval
    assert not review("click", {"index": 1}, el("Next")).needs_approval
    assert not review("click", {"index": 1}, el("Bookmarks")).needs_approval  # word boundaries

    pw = el("Password", kind="input[password]", sensitive="password")
    assert "never enters passwords" in review("type_text", {"index": 1, "text": "x"}, pw).blocked

    assert review("type_text", {"index": 1, "text": "2", "press_enter": True}, el("Qty", "input[text]"),
                  form_submit_text="Buy now").needs_approval
    assert not review("type_text", {"index": 1, "text": "q", "press_enter": True}, el("Q", "input[text]"),
                      form_submit_text="Search").needs_approval
    assert review("upload_file", {"index": 1, "filename": "cv.pdf"}, el("CV", "file")).needs_approval
    assert review("handle_next_dialog", {"accept": True}, None).needs_approval
    assert not review("handle_next_dialog", {"accept": False}, None).needs_approval


def test_domain_allowed():
    allowed = ["example.com", "wikipedia.org"]
    assert domain_allowed("https://example.com/a", allowed)
    assert domain_allowed("https://en.wikipedia.org/wiki/X", allowed)  # subdomain
    assert not domain_allowed("https://notexample.com/", allowed)  # suffix trick
    assert not domain_allowed("https://example.com.evil.io/", allowed)
    assert domain_allowed("about:blank", allowed)
    assert domain_allowed("https://anything.io", [])  # no allowlist


def test_cost_estimate():
    u = Usage(input_tokens=1_000_000, output_tokens=100_000, cache_read_tokens=1_000_000, cache_write_tokens=0)
    assert u.cost_usd("claude-opus-5-5") == pytest.approx(4.0 + 2.0 + 0.2)
    assert u.cost_usd("some-unknown-model") is None


# ---------- in the browser ----------

@pytest.fixture
async def make(server):
    sessions = []

    async def _make(mode="deny", approver=None, allowed=("127.0.0.1",)):
        session = BrowserSession(Settings(headless=True, allowed_domains=tuple(allowed)))
        await session.start()
        sessions.append(session)
        ex = ToolExecutor(session, SafetyGuard(mode=mode, approver=approver))
        await session.page.goto(server + "/safety.html", wait_until="load")
        ex.state = await DomExtractor().snapshot(session.page)
        return ex

    yield _make
    for s in sessions:
        await s.close()


def idx(ex, name):
    return next(e.index for e in ex.state.elements if e.name == name)


async def out(ex):
    return await ex.page.inner_text("#out")


async def test_sensitive_fields_flagged_and_blocked(make):
    ex = await make()
    kinds = {e.name: e.sensitive for e in ex.state.elements}
    assert kinds["Password"] == "password"
    assert kinds["Card number"] == "payment"
    assert kinds["Verification code"] == "otp"
    assert kinds["Username"] is None
    assert "password field - user only" in ex.state.to_prompt()

    for field in ("Password", "Card number", "Verification code"):
        r = await ex.execute("type_text", {"index": idx(ex, field), "text": "1234"})
        assert r.is_error and "Not typed" in r.text
    assert await ex.page.input_value("#pw") == ""
    assert not (await ex.execute("type_text", {"index": idx(ex, "Username"), "text": "roi"})).is_error
    assert any("blocked" in e for e in ex.guard.log)


async def test_risky_click_refused_without_approval(make):
    ex = await make(mode="deny")
    r = await ex.execute("click", {"index": idx(ex, "Delete account")})
    assert r.is_error and "needs the user's approval" in r.text
    assert await out(ex) == "idle"  # nothing happened
    assert not (await ex.execute("click", {"index": idx(ex, "Refresh")})).is_error  # harmless click OK
    assert await out(ex) == "refreshed"


async def test_human_approval_flow(make):
    asked = []

    async def approver(description):
        asked.append(description)
        return "Delete" not in description  # user says no to deleting, yes to buying

    ex = await make(mode="ask", approver=approver)
    r = await ex.execute("click", {"index": idx(ex, "Delete account")})
    assert r.is_error and "declined" in r.text and await out(ex) == "idle"

    r = await ex.execute("type_text", {"index": idx(ex, "Quantity"), "text": "2", "press_enter": True})
    assert not r.is_error and await out(ex) == "bought"
    assert len(asked) == 2 and "Buy now" in asked[1]
    assert [e["approved"] for e in ex.guard.log] == [False, True]

    # Enter in a search form needs no approval at all.
    await ex.execute("type_text", {"index": idx(ex, "Search"), "text": "shoes", "press_enter": True})
    assert len(asked) == 2 and await out(ex) == "searched"


async def test_unanswerable_prompt_counts_as_no(make):
    async def broken(description):
        raise EOFError  # e.g. stdin closed

    ex = await make(mode="ask", approver=broken)
    r = await ex.execute("click", {"index": idx(ex, "Delete account")})
    assert r.is_error and "Not performed" in r.text and await out(ex) == "idle"
    assert ex.guard.mode == "deny"  # no further prompts this run
    assert "approvals are disabled" in (await ex.execute("click", {"index": idx(ex, "Delete account")})).text


async def test_domain_allowlist(make):
    ex = await make()
    start = ex.page.url

    r = await ex.execute("navigate", {"url": "https://evil.example/"})
    assert r.is_error and "outside the allowed domains" in r.text

    await ex.execute("click", {"index": idx(ex, "Claim your prize")})  # link to another domain
    await ex.session.settle()
    assert ex.page.url == start
    assert any("blocked: outside the allowed domains" in e for e in ex.session.drain_events())

    await ex.execute("click", {"index": idx(ex, "Internal page")})  # same domain is fine
    await ex.session.settle()
    assert ex.page.url.endswith("/spa.html")


async def test_popup_to_other_domain_blocked(make):
    ex = await make()
    await ex.execute("click", {"index": idx(ex, "Open partner site")})
    await ex.session.settle()
    await ex.page.wait_for_timeout(300)
    assert any("evil.example" in e and "blocked" in e for e in ex.session.drain_events())
    assert all("evil.example" not in p.url for p in ex.session.pages)
    assert len(ex.session.pages) == 1  # the blank pop-up was closed
    assert ex.page.url.endswith("/safety.html")  # and we are back on the original tab


# ---------- run limits ----------

class FakeLLM:
    def __init__(self, script):
        self.script = list(script)

    async def create(self, messages, **kwargs):
        name, args = self.script.pop(0)
        block = SimpleNamespace(type="tool_use", id=f"t{len(self.script)}", name=name, input=args)
        return SimpleNamespace(content=[block], stop_reason="tool_use", context_management=None,
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=100,
                                                     cache_read_input_tokens=0, cache_creation_input_tokens=0))


async def test_step_reminder_and_budget_stop(server):
    async with BrowserSession(Settings(headless=True)) as session:
        await session.page.goto(server + "/safety.html", wait_until="load")
        refresh = ("scroll", {"direction": "down"})

        agent = Agent("t", session, FakeLLM([refresh] * 3 + [("done", {"success": True, "answer": "x"})]),
                      Settings(headless=True, max_steps=6, vision=False, trace_dir=""))
        await agent.run()
        texts = [b.get("text", "") for m in agent.messages if m["role"] == "user" and isinstance(m["content"], list)
                 for b in m["content"] if b.get("type") == "text"]
        assert any("5 steps left" in t for t in texts)

        agent = Agent("t", session, FakeLLM([refresh] * 5),
                      Settings(headless=True, max_steps=10, vision=False, trace_dir="", max_cost_usd=0.0))
        result = await agent.run()
        assert result.stop_reason == "budget" and not result.success
        assert result.steps == 1 and result.cost_usd > 0

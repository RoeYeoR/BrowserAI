"""The agent loop: observe -> think -> act -> verify -> repeat.

* observe: DOM snapshot + annotated screenshot, captured concurrently
* think:   Claude picks tool calls
* act:     tools run in order against the live page
* verify:  the new state is diffed against the old one; "nothing changed" is reported
           explicitly so the model notices failed actions instead of assuming success

Conversation history is strictly append-only: every assistant response (including its
thinking blocks) is sent back unchanged. That is required for the model's thinking to stay
valid across turns, and it keeps the prompt cache warm. Old page states and screenshots are
trimmed by the API's server-side context editing instead of by rewriting history here.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Error as PlaywrightError

from browserai.agent.llm import LLMClient
from browserai.agent.safety import SafetyGuard
from browserai.agent.prompts import SYSTEM_PROMPT, first_user_message
from browserai.agent.tools import ELEMENT_TOOLS, READ_ONLY_TOOLS, ToolExecutor, tool_definitions
from browserai.browser.session import BrowserSession
from browserai.config import Settings
from browserai.perception.dom import DomExtractor
from browserai.perception.state import PageState
from browserai.perception.vision import annotate, image_block
from browserai.tracing import Tracer

_CONTEXT_BETA = "context-management-2025-06-27"
_UPDATES_BETA = "thinking-display-updates-2026-08-18"
# Models that return short progress notes between tool calls with display="updates".
_UPDATES_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5")

# Clear old tool results (page states, screenshots, page text) server-side once the prompt
# grows, keeping the latest few. Notes and the final answer are never cleared.
_CONTEXT_MANAGEMENT = {
    "edits": [
        {
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": 50_000},
            "keep": {"type": "tool_uses", "value": 3},
            "clear_at_least": {"type": "input_tokens", "value": 10_000},
            "exclude_tools": ["save_note", "done"],
        }
    ]
}

MAX_NUDGES = 2
NO_CHANGE_WARNING = (
    "⚠ No change detected: URL, elements, values and scroll position are the same as before. "
    "The action probably had no effect - check the screenshot and try a different approach "
    "rather than repeating it."
)


@dataclass
class Event:
    kind: str  # "step" | "thought" | "tool_call" | "tool_result" | "verify" | "context" | "done"
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, u: Any) -> None:
        self.input_tokens += u.input_tokens or 0
        self.output_tokens += u.output_tokens or 0
        self.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0

    def cost_usd(self, model: str) -> float | None:
        """Estimated spend from list prices ($/MTok); None for models not in the table."""
        prices = next((p for prefix, p in PRICES.items() if model.startswith(prefix)), None)
        if prices is None:
            return None
        inp, out, read, write = prices
        return (self.input_tokens * inp + self.output_tokens * out
                + self.cache_read_tokens * read + self.cache_write_tokens * write) / 1_000_000


# input, output, cache read, cache write (5-minute writes cost 1.25x input)
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}


@dataclass
class RunResult:
    success: bool
    answer: str
    steps: int
    stop_reason: str  # "done" | "max_steps" | "no_action" | "budget"
    notes: list[str]
    usage: Usage
    seconds: float
    trace_dir: str | None = None
    cost_usd: float | None = None
    safety_log: list[dict[str, Any]] = field(default_factory=list)  # approvals, refusals, blocks


@dataclass
class Observation:
    state: PageState
    content: list[dict[str, Any]]  # text (+ image) blocks for the model
    screenshot: bytes | None


class Agent:
    def __init__(
        self,
        task: str,
        session: BrowserSession,
        llm: LLMClient,
        settings: Settings,
        on_event: Callable[[Event], None] | None = None,
        guard: SafetyGuard | None = None,
    ) -> None:
        self.task = task
        self.session = session
        self.llm = llm
        self.settings = settings
        self.extractor = DomExtractor()
        self.tools = ToolExecutor(session, guard)
        self.messages: list[dict[str, Any]] = []
        self.usage = Usage()
        self._listeners = [on_event] if on_event else []
        self.tracer = (
            Tracer(settings.trace_dir, task, {"model": settings.model, "effort": settings.effort,
                                              "vision": settings.vision})
            if settings.trace_dir else None
        )
        self._record: dict[str, Any] = {}

    # ---------- events / tracing ----------

    def _emit(self, kind: str, **data: Any) -> None:
        event = Event(kind, data)
        for listener in self._listeners:
            listener(event)
        if kind in ("thought", "tool_call", "tool_result", "verify", "context"):
            self._record.setdefault("events", []).append({"kind": kind, **data})

    def _flush_record(self) -> None:
        if self.tracer and self._record:
            self.tracer.step(self._record)
        self._record = {}

    # ---------- observe ----------

    async def _observe(self, step: int) -> Observation:
        page = self.session.page
        if self.settings.vision:
            # DOM extraction and the screenshot are independent - capture them concurrently.
            state, png = await asyncio.gather(
                self.extractor.snapshot(page),
                page.screenshot(type="png", animations="disabled"),
            )
            jpeg = await asyncio.to_thread(annotate, png, state)  # Pillow work off the event loop
        else:
            state, jpeg = await self.extractor.snapshot(page), None

        self.tools.state = state
        text = state.to_prompt()
        tabs = await self.session.tabs_text()
        if tabs:
            text = tabs + "\n\n" + text
        if jpeg:
            text += "\n\n(Screenshot of the viewport attached; numbered boxes match the element numbers.)"
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if jpeg:
            content.append(image_block(jpeg))
            if self.tracer:
                self._record["screenshot"] = self.tracer.screenshot(step, jpeg)
        return Observation(state, content, jpeg)

    # ---------- main loop ----------

    def _request_options(self) -> dict[str, Any]:
        betas = [_CONTEXT_BETA]
        opts: dict[str, Any] = {"context_management": _CONTEXT_MANAGEMENT}
        if self.settings.model.startswith(_UPDATES_MODELS):
            betas.append(_UPDATES_BETA)
            opts["thinking"] = {"type": "adaptive", "display": "updates"}
        # Automatic prompt caching: the static system prompt + tools + growing history prefix.
        opts["cache_control"] = {"type": "ephemeral"}
        return {"betas": betas, **opts}

    async def run(self) -> RunResult:
        t0 = time.perf_counter()
        try:
            result = await self._run(t0)
        finally:
            self._flush_record()
        if self.tracer:
            result.trace_dir = str(self.tracer.dir)
            self.tracer.finish(result)
        return result

    async def _run(self, t0: float) -> RunResult:
        await self.session.settle()
        self._record = {"step": 0, "url": self.session.page.url}
        obs = await self._observe(0)
        self._flush_record()
        first = obs.content
        first[0] = {"type": "text", "text": first_user_message(self.task, first[0]["text"])}
        self.messages.append({"role": "user", "content": first})

        tools = tool_definitions()
        options = self._request_options()
        nudges = 0

        for step in range(1, self.settings.max_steps + 1):
            self._record = {"step": step, "url": self.session.page.url}
            self._emit("step", step=step, url=self.session.page.url)
            response = await self.llm.create(self.messages, system=SYSTEM_PROMPT, tools=tools, **options)
            self.usage.add(response.usage)
            self._record["stop_reason"] = response.stop_reason
            # Append the assistant turn exactly as received (thinking blocks included).
            self.messages.append({"role": "assistant", "content": response.content})
            self._report_context(response)

            tool_uses = []
            for block in response.content:
                if block.type == "thinking" and block.thinking:
                    self._emit("thought", text=block.thinking)
                elif block.type == "text" and block.text.strip():
                    self._emit("thought", text=block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if response.stop_reason == "max_tokens":
                self._append_cutoff(tool_uses)
                self._flush_record()
                continue

            if not tool_uses:
                nudges += 1
                if nudges > MAX_NUDGES:
                    text = " ".join(b.text for b in response.content if b.type == "text").strip()
                    return self._result(False, text or "Agent stopped without an answer.", step, "no_action", t0)
                self.messages.append({
                    "role": "user",
                    "content": "Continue the task by calling a tool. Call `done` when finished.",
                })
                self._flush_record()
                continue
            nudges = 0

            results, done = await self._execute(tool_uses, step)
            if done is not None:
                return self._result(done.success, done.answer, step, "done", t0)
            self._add_budget_notes(results, step)
            self.messages.append({"role": "user", "content": results})
            cost = self.usage.cost_usd(self.settings.model)
            if cost is not None and cost > self.settings.max_cost_usd:
                return self._result(False, f"Stopped: estimated cost ${cost:.2f} exceeded the "
                                    f"${self.settings.max_cost_usd:.2f} limit for this run.", step, "budget", t0)
            self._flush_record()

        return self._result(False, f"Stopped after {self.settings.max_steps} steps without finishing.",
                            self.settings.max_steps, "max_steps", t0)

    # ---------- act + verify ----------

    async def _execute(self, tool_uses: list[Any], step: int) -> tuple[list[dict[str, Any]], Any]:
        """Run tool calls in order. Every tool_use gets a tool_result, all in one user message."""
        results: list[dict[str, Any]] = []
        before = self.tools.state
        start_page = self.session.page
        start_url = start_page.url
        page_changed = False
        acted = False

        for tu in tool_uses:
            self._emit("tool_call", name=tu.name, input=tu.input)
            if page_changed and tu.name in ELEMENT_TOOLS:
                # Element numbers came from the old page; acting on them could hit the wrong thing.
                text = "skipped: an earlier action in this turn changed the page. Re-check the new page state."
                results.append(self._tool_result(tu.id, text, is_error=True))
                self._emit("tool_result", name=tu.name, text=text, is_error=True)
                continue

            outcome = await self.tools.execute(tu.name, tu.input)
            self._emit("tool_result", name=tu.name, text=outcome.text, is_error=outcome.is_error)
            if outcome.done is not None:
                self._emit("done", success=outcome.done.success, answer=outcome.done.answer)
                return results, outcome.done
            results.append(self._tool_result(tu.id, outcome.text, outcome.is_error))

            if tu.name not in READ_ONLY_TOOLS:
                acted = True
                await self._settle()
                if self.session.page is not start_page or self.session.page.url != start_url:
                    page_changed = True

        if acted:
            obs = await self._observe(step)
            feedback = self._verify(before, obs.state, new_tab=self.session.page is not start_page)
            # Attach verification + fresh page state (+ screenshot) to a tool result, so that
            # context editing can clear it later like any other old result.
            target = self._observation_target(results)
            target["content"].append({"type": "text", "text": feedback + "\n\nPage state now:"})
            target["content"].extend(obs.content)
        return results, None

    @staticmethod
    def _observation_target(results: list[dict[str, Any]]) -> dict[str, Any]:
        """The API only allows text inside an is_error tool result, so the screenshot goes on the
        last successful result. If every call failed, the last one carries it, with its error
        stated in the text instead of the is_error flag."""
        for block in reversed(results):
            if not block.get("is_error"):
                return block
        last = results[-1]
        last.pop("is_error", None)
        first = last["content"][0]
        first["text"] = "ERROR: " + first["text"]
        return last

    async def _settle(self) -> None:
        try:
            await self.session.settle()
        except PlaywrightError:
            pass  # the page/tab closed mid-wait; the next snapshot will show where we are

    def _verify(self, before: PageState | None, after: PageState, *, new_tab: bool) -> str:
        changes = after.diff(before)
        # Browser events the DOM can't show: dialogs, new/closed tabs, blocked downloads.
        changes[:0] = self.session.drain_events()
        if changes:
            self._emit("verify", changed=True, changes=changes)
            return "Changes since your last action:\n" + "\n".join(f"- {c}" for c in changes)
        self._emit("verify", changed=False, changes=[])
        return NO_CHANGE_WARNING

    @staticmethod
    def _tool_result(tool_use_id: str, text: str, is_error: bool = False) -> dict[str, Any]:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": [{"type": "text", "text": text}],
        }
        if is_error:
            block["is_error"] = True
        return block

    def _append_cutoff(self, tool_uses: list[Any]) -> None:
        msg = "Your response was cut off by the output limit. Re-issue the action with shorter input."
        if tool_uses:
            self.messages.append({"role": "user", "content": [self._tool_result(t.id, msg, True) for t in tool_uses]})
        else:
            self.messages.append({"role": "user", "content": msg})

    def _report_context(self, response: Any) -> None:
        cm = getattr(response, "context_management", None)
        for edit in getattr(cm, "applied_edits", None) or []:
            self._emit("context",
                       cleared_tool_uses=getattr(edit, "cleared_tool_uses", None),
                       cleared_input_tokens=getattr(edit, "cleared_input_tokens", None))

    def _add_budget_notes(self, results: list[dict[str, Any]], step: int) -> None:
        """Warn the model as it approaches the step limit. Added as text after the tool results
        (append-only, so the conversation prefix and cache stay intact)."""
        left = self.settings.max_steps - step
        if left in (5, 2):
            results.append({"type": "text", "text": (
                f"(Note: {left} steps left before this run is stopped. Wrap up: save what you "
                "found and call `done`, with success=false if the task can't be finished in time.)"
            )})

    def _result(self, success: bool, answer: str, steps: int, reason: str, t0: float) -> RunResult:
        return RunResult(success, answer, steps, reason, list(self.tools.notes), self.usage,
                         time.perf_counter() - t0, cost_usd=self.usage.cost_usd(self.settings.model),
                         safety_log=list(self.tools.guard.log))

"""Tool definitions the model can call, and the executor that runs them.

Each tool's input is a pydantic model: the JSON schema sent to Claude is generated from it,
and the model's arguments are validated against it before anything touches the browser.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from browserai.agent.safety import SafetyGuard, domain_allowed, review
from browserai.browser import actions, forms
from browserai.browser.actions import ActionError
from browserai.browser.session import BrowserSession
from browserai.perception.state import Element, PageState


# ---------- tool inputs ----------

class Navigate(BaseModel):
    """Open a URL in the current tab."""
    url: str = Field(description="Absolute URL, e.g. https://example.com/path")


class Click(BaseModel):
    """Click an element from the current page state."""
    index: int = Field(description="Element number, e.g. 12 for [12]")


class TypeText(BaseModel):
    """Type into a text field, textarea or editable element."""
    index: int = Field(description="Element number of the field")
    text: str = Field(description="Text to type")
    clear: bool = Field(True, description="Replace the current value (true) or append (false)")
    press_enter: bool = Field(False, description="Press Enter after typing, e.g. to submit a search")


class SelectOption(BaseModel):
    """Choose an option in a dropdown: a native <select>, or a custom combobox/listbox/autocomplete
    (it is opened and the matching option clicked)."""
    index: int
    option: str = Field(description="Visible option text (case-insensitive; a unique prefix or part is enough)")


class SetCheckbox(BaseModel):
    """Check or uncheck a checkbox or switch, or select a radio button. Verifies the new state."""
    index: int
    checked: bool = True


class SetDate(BaseModel):
    """Fill a date field: native date/month/time inputs, or a text field behind a custom date picker."""
    index: int
    date: str = Field(description="ISO date, YYYY-MM-DD (or YYYY-MM-DDTHH:MM)")
    display_format: str | None = Field(
        None, description="Only for custom text fields, e.g. DD/MM/YYYY; defaults to the field's placeholder"
    )


class UploadFile(BaseModel):
    """Attach a file from the uploads folder to a file input (or an upload button)."""
    index: int
    filename: str = Field(description="File name inside the uploads folder")


class Scroll(BaseModel):
    """Scroll the page, or a scrollable container if `index` is given."""
    direction: Literal["up", "down"]
    amount: float = Field(0.8, ge=0.1, le=5, description="Distance in viewport heights")
    index: int | None = Field(None, description="Optional element inside the container to scroll")


class PressKey(BaseModel):
    """Press a keyboard key or shortcut on the focused element."""
    key: str = Field(description="Playwright key name, e.g. Enter, Escape, Tab, ArrowDown, Control+A")


class GoBack(BaseModel):
    """Go back to the previous page in this tab's history."""


class Wait(BaseModel):
    """Wait for slow content to appear."""
    seconds: float = Field(2, ge=0.5, le=10)


class SwitchTab(BaseModel):
    """Make another open tab the active one (tab numbers are shown under "Open tabs")."""
    tab: int = Field(ge=0)


class CloseTab(BaseModel):
    """Close the active tab and return to the previous one."""


class HandleNextDialog(BaseModel):
    """Choose how to answer the NEXT confirm()/prompt() dialog (they are cancelled by default).
    Call this, then repeat the action that opens the dialog."""
    accept: bool
    prompt_text: str | None = Field(None, description="Text to enter if the dialog is a prompt()")


class ReadText(BaseModel):
    """Read the visible text content of the page (articles, prices, tables, search results)."""
    max_chars: int = Field(6000, ge=500, le=20000)


class SaveNote(BaseModel):
    """Save a fact you will need later (e.g. a price you found). Notes are never cleared from
    context, unlike old page states, so record anything needed for the final answer."""
    note: str


class Done(BaseModel):
    """Finish the task. Call this exactly once, when the task is complete or impossible."""
    success: bool = Field(description="Whether the task was accomplished")
    answer: str = Field(description="Final answer / summary for the user, including any requested data")


TOOLS: dict[str, type[BaseModel]] = {
    "navigate": Navigate,
    "click": Click,
    "type_text": TypeText,
    "select_option": SelectOption,
    "set_checkbox": SetCheckbox,
    "set_date": SetDate,
    "upload_file": UploadFile,
    "scroll": Scroll,
    "press_key": PressKey,
    "go_back": GoBack,
    "wait": Wait,
    "switch_tab": SwitchTab,
    "close_tab": CloseTab,
    "handle_next_dialog": HandleNextDialog,
    "read_text": ReadText,
    "save_note": SaveNote,
    "done": Done,
}

# Tools whose effect depends on element numbers from the latest page state.
ELEMENT_TOOLS = {"click", "type_text", "select_option", "set_checkbox", "set_date", "upload_file"}
# Tools that never change the page, so no fresh snapshot is needed afterwards.
READ_ONLY_TOOLS = {"read_text", "save_note", "done", "handle_next_dialog"}


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.pop("description", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    schema.setdefault("properties", {})
    return schema


def tool_definitions() -> list[dict[str, Any]]:
    """Tool list for the Messages API, in a fixed order (keeps the prompt cache stable)."""
    return [
        {"name": name, "description": (model.__doc__ or "").strip(), "input_schema": _schema(model)}
        for name, model in TOOLS.items()
    ]


# ---------- execution ----------

@dataclass
class ToolOutcome:
    text: str
    is_error: bool = False
    done: Done | None = None


_READ_TEXT_JS = """
(maxChars) => {
  const root = document.querySelector('main, [role=main], article') || document.body;
  const text = (root ? root.innerText : '').replace(/\\n{3,}/g, '\\n\\n').trim();
  return text.length > maxChars ? text.slice(0, maxChars) + '\\n…(truncated)' : text;
}
"""


def _refound_note(el: Element, text: str) -> str:
    if el.refound:
        el.refound = False
        return text + " (the page had re-rendered it; re-found it by its kind and name)"
    return text


_FORM_SUBMIT_TEXT_JS = """
e => {
  const f = e.form || e.closest('form');
  if (!f) return '';
  const b = f.querySelector('button[type=submit], button:not([type]), input[type=submit]');
  return b ? (b.innerText || b.value || '').trim() : '';
}
"""


class ToolExecutor:
    def __init__(self, session: BrowserSession, guard: SafetyGuard | None = None) -> None:
        self.session = session
        # Default: risky actions are refused unless a guard with approvals is supplied.
        self.guard = guard or SafetyGuard(mode="deny")
        self.state: PageState | None = None  # latest snapshot; element lookups use it
        self.notes: list[str] = []

    async def execute(self, name: str, raw_input: dict[str, Any]) -> ToolOutcome:
        model = TOOLS.get(name)
        if model is None:
            return ToolOutcome(f"unknown tool {name!r}", is_error=True)
        try:
            args = model.model_validate(raw_input)
        except ValidationError as e:
            return ToolOutcome(f"invalid arguments for {name}: {e.errors(include_url=False)}", is_error=True)
        try:
            refusal = await self._safety_check(name, args)
            if refusal:
                return ToolOutcome(refusal, is_error=True)
            return await getattr(self, f"_{name}")(args)
        except ActionError as e:
            return ToolOutcome(str(e), is_error=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # never let one bad action kill the run; tell the model instead
            return ToolOutcome(f"{name} failed: {type(e).__name__}: {str(e).splitlines()[0][:200]}", is_error=True)

    async def _safety_check(self, name: str, args: BaseModel) -> str | None:
        """Return a refusal message, or None if the action may go ahead."""
        index = getattr(args, "index", None)
        el = self.state.get(index) if (self.state and index is not None) else None
        submit_text = ""
        if name == "type_text" and el is not None and args.press_enter:
            try:
                submit_text = await actions.locate(el).first.evaluate(_FORM_SUBMIT_TEXT_JS)
            except Exception:
                submit_text = ""
        verdict = review(name, args.model_dump(), el, submit_text)
        if verdict.blocked:
            self.guard.log.append({"action": f"{name} {args.model_dump()}", "blocked": verdict.blocked})
            return verdict.blocked
        if verdict.needs_approval and not await self.guard.approve(verdict.needs_approval):
            return self.guard.refusal(verdict.needs_approval)
        return None

    def _element(self, index: int) -> Element:
        el = self.state.get(index) if self.state else None
        if el is None:
            raise ActionError(f"there is no element [{index}] on the current page. Use the latest element list.")
        return el

    @property
    def page(self):
        return self.session.page

    async def _navigate(self, a: Navigate) -> ToolOutcome:
        url = a.url.strip()
        if not url.startswith(("http://", "https://")):
            if "://" in url:
                raise ActionError("only http(s) URLs are allowed")
            url = "https://" + url
        allowed = self.session.settings.allowed_domains
        if not domain_allowed(url, allowed):
            raise ActionError(f"{url} is outside the allowed domains for this run ({', '.join(allowed)})")
        await self.session.goto(url)
        return ToolOutcome(f"navigated to {self.page.url}")

    async def _click(self, a: Click) -> ToolOutcome:
        el = self._element(a.index)
        return ToolOutcome(_refound_note(el, await actions.click(el)))

    async def _type_text(self, a: TypeText) -> ToolOutcome:
        el = self._element(a.index)
        text = await actions.type_text(el, a.text, clear=a.clear, press_enter=a.press_enter)
        return ToolOutcome(_refound_note(el, text))

    async def _select_option(self, a: SelectOption) -> ToolOutcome:
        el = self._element(a.index)
        return ToolOutcome(_refound_note(el, await forms.select_option(el, a.option)))

    async def _set_checkbox(self, a: SetCheckbox) -> ToolOutcome:
        el = self._element(a.index)
        return ToolOutcome(_refound_note(el, await forms.set_checked(el, a.checked)))

    async def _set_date(self, a: SetDate) -> ToolOutcome:
        el = self._element(a.index)
        return ToolOutcome(_refound_note(el, await forms.set_date(el, a.date, a.display_format)))

    async def _upload_file(self, a: UploadFile) -> ToolOutcome:
        el = self._element(a.index)
        return ToolOutcome(await forms.upload_file(el, a.filename, self.session.settings.upload_dir))

    async def _switch_tab(self, a: SwitchTab) -> ToolOutcome:
        try:
            page = await self.session.switch_tab(a.tab)
        except IndexError as e:
            raise ActionError(str(e))
        return ToolOutcome(f"switched to tab {a.tab}: {page.url}")

    async def _close_tab(self, a: CloseTab) -> ToolOutcome:
        try:
            await self.session.close_tab()
        except RuntimeError as e:
            raise ActionError(str(e))
        return ToolOutcome(f"closed the tab; now on {self.page.url}")

    async def _handle_next_dialog(self, a: HandleNextDialog) -> ToolOutcome:
        self.session.next_dialog = (a.accept, a.prompt_text)
        return ToolOutcome(f"the next dialog will be {'accepted' if a.accept else 'cancelled'}; now repeat the action")

    async def _scroll(self, a: Scroll) -> ToolOutcome:
        el = self._element(a.index) if a.index is not None else None
        return ToolOutcome(await actions.scroll(self.page, a.direction, a.amount, el))

    async def _press_key(self, a: PressKey) -> ToolOutcome:
        await self.page.keyboard.press(a.key)
        return ToolOutcome(f"pressed {a.key}")

    async def _go_back(self, a: GoBack) -> ToolOutcome:
        resp = await self.page.go_back(wait_until="domcontentloaded")
        if resp is None and self.page.url == (self.state.url if self.state else None):
            raise ActionError("there is no previous page in this tab's history")
        return ToolOutcome(f"went back to {self.page.url}")

    async def _wait(self, a: Wait) -> ToolOutcome:
        await asyncio.sleep(a.seconds)
        return ToolOutcome(f"waited {a.seconds:g}s")

    async def _read_text(self, a: ReadText) -> ToolOutcome:
        text = await self.page.evaluate(_READ_TEXT_JS, a.max_chars)
        return ToolOutcome(f"Page text ({self.page.url}):\n{text or '(no text)'}")

    async def _save_note(self, a: SaveNote) -> ToolOutcome:
        self.notes.append(a.note)
        return ToolOutcome(f"noted ({len(self.notes)} notes so far)")

    async def _done(self, a: Done) -> ToolOutcome:
        return ToolOutcome("finished", done=a)

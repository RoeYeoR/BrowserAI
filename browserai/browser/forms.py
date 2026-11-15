"""Form-control actions: dropdowns (native and custom), checkboxes/switches, dates, file upload.

Every action verifies its result (re-reads the control's value/state) instead of assuming
the click or fill worked, and errors explain what the model can do next.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
from pathlib import Path

from playwright.async_api import Error as PlaywrightError

from browserai.browser.actions import CLICK_TIMEOUT_MS, ActionError, _ensure_attached, _short_reason
from browserai.perception.state import Element

_OPTION_SELECTOR = "[role=option], [role=menuitemradio], [role=menuitem], [role=treeitem]"


def best_match(query: str, candidates: list[str]) -> int | None:
    """Index of the candidate matching `query`: exact (case-insensitive), then unique prefix,
    then unique substring. None if nothing or several things match."""
    q = " ".join(query.lower().split())
    if not q:
        return None
    norm = [" ".join(c.lower().split()) for c in candidates]
    exact = [i for i, c in enumerate(norm) if c == q]
    if exact:
        return exact[0]  # duplicate labels are the same choice for the user
    for test in (lambda c: c.startswith(q), lambda c: q in c):
        hits = [i for i, c in enumerate(norm) if test(c)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return None  # ambiguous: make the model be specific
    return None


def _options_list(options: list[str], limit: int = 20) -> str:
    shown = ", ".join(f'"{o}"' for o in options[:limit])
    return shown + (f" …(+{len(options) - limit})" if len(options) > limit else "")


# ---------- dropdowns ----------

async def select_option(el: Element, option: str) -> str:
    loc = await _ensure_attached(el)

    if el.tag == "select":
        opts = await loc.evaluate(
            "e => Array.from(e.options).map(o => ({label: o.text.trim(), value: o.value, disabled: o.disabled}))"
        )
        labels = [o["label"] for o in opts]
        i = best_match(option, labels)
        if i is None:
            i = best_match(option, [o["value"] for o in opts])
        if i is None:
            raise ActionError(f'no single option matches "{option}" in [{el.index}]. Options: {_options_list(labels)}')
        if opts[i]["disabled"]:
            raise ActionError(f'option "{labels[i]}" in [{el.index}] is disabled')
        await loc.select_option(value=opts[i]["value"], timeout=CLICK_TIMEOUT_MS)
        return f'selected "{labels[i]}" in [{el.index}]'

    # Custom widget (ARIA combobox / listbox button / autocomplete input): open it, then pick.
    frame = el.frame
    if el.tag == "input" or el.kind == "editable":
        await loc.fill(option, timeout=CLICK_TIMEOUT_MS)  # autocomplete: typing filters the list
    else:
        await loc.click(timeout=CLICK_TIMEOUT_MS)

    controls = await loc.get_attribute("aria-controls") or await loc.get_attribute("aria-owns")
    scope = frame.locator(f"#{controls}") if controls and re.fullmatch(r"[\w-]+", controls) else frame
    options = scope.locator(_OPTION_SELECTOR).locator("visible=true")

    for _ in range(15):  # options often render asynchronously (fetch-backed autocompletes)
        if await options.count():
            break
        await asyncio.sleep(0.15)
    count = await options.count()
    if not count:
        raise ActionError(
            f"opened [{el.index}] but no options appeared. Look at the new page state and "
            "click the option you need directly."
        )
    texts = [" ".join(t.split()) for t in await options.all_inner_texts()]
    i = best_match(option, texts)
    if i is None:
        raise ActionError(f'no single option matches "{option}". Visible options: {_options_list(texts)}')
    await options.nth(i).click(timeout=CLICK_TIMEOUT_MS)
    return f'opened [{el.index}] and chose "{texts[i]}"'


# ---------- checkboxes / radios / switches ----------

async def set_checked(el: Element, checked: bool) -> str:
    if el.kind not in ("checkbox", "radio", "switch", "menuitemcheckbox", "menuitemradio"):
        raise ActionError(f"[{el.index}] is a {el.kind}, not a checkbox, radio button or switch")
    if el.kind in ("radio", "menuitemradio") and not checked:
        raise ActionError("a radio button can't be unchecked directly; select another option in its group")
    if el.disabled:
        raise ActionError(f"[{el.index}] is disabled")
    loc = await _ensure_attached(el)

    read_state = (
        "e => e.tagName === 'INPUT' ? e.checked : e.getAttribute('aria-checked') === 'true'"
    )
    if await loc.evaluate(read_state) == checked:
        return f"[{el.index}] was already {'checked' if checked else 'unchecked'}"

    if el.via_label:
        await loc.evaluate("e => (e.labels && e.labels[0] ? e.labels[0] : e).click()")
    else:
        try:
            await loc.click(timeout=CLICK_TIMEOUT_MS)
        except PlaywrightError:
            await loc.evaluate("e => e.click()")

    if await loc.evaluate(read_state) != checked:
        raise ActionError(f"clicked [{el.index}] but its state did not change - the page may block it")
    return f"[{el.index}] is now {'checked' if checked else 'unchecked'}"


# ---------- dates ----------

_NATIVE_DATE_FORMATS = {
    "date": "%Y-%m-%d",
    "month": "%Y-%m",
    "datetime-local": "%Y-%m-%dT%H:%M",
    "time": "%H:%M",
}


def format_date(value: dt.datetime, pattern: str) -> str:
    """Format with a human pattern such as DD/MM/YYYY, MM-DD-YY or YYYY.MM.DD."""
    has_time = "HH" in pattern or "hh" in pattern
    codes = {
        "YYYY": "%Y", "yyyy": "%Y", "YY": "%y", "yy": "%y",
        "MM": "%m", "DD": "%d", "dd": "%d", "HH": "%H", "hh": "%H",
        "mm": "%M" if has_time else "%m",  # "mm" is minutes only next to an hour token
    }
    return re.sub(r"YYYY|yyyy|YY|yy|MM|DD|dd|HH|hh|mm", lambda m: value.strftime(codes[m.group()]), pattern)


async def set_date(el: Element, date: str, display_format: str | None = None) -> str:
    try:
        value = dt.datetime.fromisoformat(date.strip())
    except ValueError:
        raise ActionError(f'"{date}" is not an ISO date; use YYYY-MM-DD (or YYYY-MM-DDTHH:MM)')
    loc = await _ensure_attached(el)
    input_type = await loc.evaluate("e => (e.type || '').toLowerCase()")

    if input_type in _NATIVE_DATE_FORMATS:
        text = value.strftime(_NATIVE_DATE_FORMATS[input_type])
        await loc.fill(text, timeout=CLICK_TIMEOUT_MS)
    else:
        # Text field behind a custom date picker: type the date in the format the field expects.
        pattern = display_format or el.placeholder or ""
        if not re.search(r"(?i)yy", pattern):
            pattern = "YYYY-MM-DD"
        text = format_date(value, pattern)
        await loc.fill(text, timeout=CLICK_TIMEOUT_MS)
        await loc.press("Tab")  # close the picker popup and let the widget parse the text

    actual = await loc.input_value()
    if not actual:
        raise ActionError(f'typed "{text}" into [{el.index}] but the field is empty - the widget rejected it; '
                          "try its calendar buttons instead")
    return f'set [{el.index}] to "{actual}"'


# ---------- file upload ----------

async def upload_file(el: Element, filename: str, upload_dir: str | Path) -> str:
    base = Path(upload_dir).resolve()
    path = (base / filename).resolve()
    if base != path.parent and base not in path.parents:
        raise ActionError(f"only files inside the uploads folder can be attached ({base})")
    if not path.is_file():
        available = sorted(p.name for p in base.glob("*") if p.is_file()) if base.is_dir() else []
        raise ActionError(f'"{filename}" not found in the uploads folder. Available: {available or "none"}')

    loc = await _ensure_attached(el)
    is_file_input = await loc.evaluate("e => e.tagName === 'INPUT' && e.type === 'file'")
    try:
        if is_file_input:
            await loc.set_input_files(str(path), timeout=CLICK_TIMEOUT_MS)
        else:
            # A styled button that opens the OS file chooser.
            async with loc.page.expect_file_chooser(timeout=5000) as info:
                await loc.click(timeout=CLICK_TIMEOUT_MS)
            await (await info.value).set_files(str(path))
    except PlaywrightError as e:
        raise ActionError(f"could not attach the file to [{el.index}]: {_short_reason(e)}")
    return f'attached "{path.name}" to [{el.index}]'

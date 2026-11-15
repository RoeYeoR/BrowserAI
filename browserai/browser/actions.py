"""Low-level browser actions on extracted elements.

Each action returns a short human-readable outcome string (it becomes part of the tool
result the model reads) and raises ActionError with an actionable message on failure.
"""

from __future__ import annotations

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Locator, Page

from browserai.perception.dom import EXTRACT_JS
from browserai.perception.state import Element

CLICK_TIMEOUT_MS = 4000
NEW_TAB_TIMEOUT_MS = 5000


class ActionError(Exception):
    pass


def locate(el: Element) -> Locator:
    if el.frame is None or el.frame.is_detached():
        raise ActionError(f"[{el.index}] belongs to an iframe that no longer exists")
    return el.frame.locator(el.selector)


async def _ensure_attached(el: Element) -> Locator:
    loc = locate(el)
    if await loc.count() > 0:
        return loc.first
    refound = await refind(el)
    if refound is None:
        raise ActionError(
            f"[{el.index}] is no longer on the page (it was removed or re-rendered). "
            "Use the element numbers from the latest page state."
        )
    return refound


async def refind(el: Element) -> Locator | None:
    """SPA frameworks often replace DOM nodes on re-render, so our tag disappears even though
    the "same" button is still there. Find the replacement - same kind and accessible name,
    nearest to the old position, and not an element we already know - and move the old tag
    onto it, so the element keeps its number for the model."""
    frame = el.frame
    try:
        prev_next = await frame.evaluate("() => (window.__bai ? window.__bai.next : 1)")
        raw = await frame.evaluate(EXTRACT_JS, {"maxElements": 2000})
    except PlaywrightError:
        return None

    ox = oy = 0.0
    if frame.parent_frame is not None:
        bb = await (await frame.frame_element()).bounding_box()
        if bb:
            ox, oy = bb["x"], bb["y"]
    cx, cy = el.box.center

    def distance(item: dict) -> float:
        b = item["box"]
        return ((b["x"] + ox + b["w"] / 2 - cx) ** 2 + (b["y"] + oy + b["h"] / 2 - cy) ** 2) ** 0.5

    candidates = [
        i for i in raw["elements"]
        if i["bai_id"] >= prev_next and i["kind"] == el.kind and i["name"] == el.name
    ]
    if not candidates:
        return None
    best = min(candidates, key=distance)
    if not el.name and distance(best) > 60:  # unnamed: only accept a near-identical position
        return None

    new_loc = frame.locator(f'[data-bai-id="{best["bai_id"]}"]').first
    await new_loc.evaluate(
        "(e, old) => { e.__baiId = old; e.setAttribute('data-bai-id', String(old)); }", el.bai_id
    )
    el.refound = True
    return locate(el).first


async def click(el: Element) -> str:
    if el.disabled:
        raise ActionError(f"[{el.index}] is disabled")
    loc = await _ensure_attached(el)

    if el.via_label:
        # Visually hidden checkbox/radio: click its label like a user would.
        await loc.evaluate("e => (e.labels && e.labels[0] ? e.labels[0] : e).click()")
        return f"clicked the label of [{el.index}]"

    if el.new_tab:
        # The new tab appears a moment after the click; wait for it so the agent's next
        # observation is of the new tab rather than the old one.
        clicked = False
        try:
            async with loc.page.context.expect_page(timeout=NEW_TAB_TIMEOUT_MS) as info:
                await loc.click(timeout=CLICK_TIMEOUT_MS)
                clicked = True
            new_page = await info.value
            try:
                await new_page.wait_for_load_state("domcontentloaded", timeout=NEW_TAB_TIMEOUT_MS)
            except PlaywrightError:
                pass
            return f"clicked [{el.index}]; it opened a new tab ({new_page.url})"
        except PlaywrightError:
            if clicked:  # never click twice
                return f"clicked [{el.index}] (it was expected to open a new tab, but none appeared)"
            # the click itself failed: fall through to the normal click + JS fallback below

    try:
        await loc.click(timeout=CLICK_TIMEOUT_MS)
        return f"clicked [{el.index}]"
    except PlaywrightError as e:
        reason = _short_reason(e)
    # Real click failed (covered, animating, zero-size...). Fall back to a DOM click event.
    try:
        await loc.evaluate("e => e.click()")
    except PlaywrightError as e:
        raise ActionError(f"could not click [{el.index}]: {reason}; JS click also failed: {_short_reason(e)}")
    return f"clicked [{el.index}] via JavaScript (normal click failed: {reason})"


async def type_text(el: Element, text: str, *, clear: bool = True, press_enter: bool = False) -> str:
    if el.disabled:
        raise ActionError(f"[{el.index}] is disabled")
    loc = await _ensure_attached(el)
    try:
        if clear:
            await loc.fill(text, timeout=CLICK_TIMEOUT_MS)
        else:
            await loc.click(timeout=CLICK_TIMEOUT_MS)
            await loc.press("End")
            await loc.press_sequentially(text)
    except PlaywrightError as e:
        # Not a fillable element (e.g. a custom widget): focus it and type with the keyboard.
        try:
            await loc.click(timeout=CLICK_TIMEOUT_MS)
            await loc.page.keyboard.type(text)
        except PlaywrightError:
            raise ActionError(f"could not type into [{el.index}] ({el.kind}): {_short_reason(e)}")
    msg = f"typed {len(text)} characters into [{el.index}]"
    if press_enter:
        await loc.press("Enter")
        msg += " and pressed Enter"
    return msg


async def scroll(page: Page, direction: str, amount: float = 0.8, el: Element | None = None) -> str:
    sign = -1 if direction == "up" else 1
    if el is not None:
        loc = await _ensure_attached(el)
        moved = await loc.evaluate(
            """(e, dy) => {
                // Scroll the element if it is a scroll container, else its nearest scrollable ancestor.
                let n = e;
                while (n && !(n.scrollHeight > n.clientHeight + 1 &&
                       /(auto|scroll)/.test(getComputedStyle(n).overflowY))) n = n.parentElement;
                if (!n) { e.scrollIntoView({block: 'center'}); return 'into-view'; }
                const before = n.scrollTop;
                n.scrollBy(0, dy * n.clientHeight);
                return n.scrollTop !== before ? 'scrolled' : 'at-edge';
            }""",
            sign * amount,
        )
        if moved == "into-view":
            return f"scrolled [{el.index}] into view"
        if moved == "at-edge":
            return f"the container of [{el.index}] is already at the {'top' if sign < 0 else 'bottom'}"
        return f"scrolled inside the container of [{el.index}] {direction}"

    vh = page.viewport_size["height"] if page.viewport_size else 800
    dy = int(sign * amount * vh)
    before = await page.evaluate("() => window.scrollY")
    await page.evaluate("dy => window.scrollBy(0, dy)", dy)
    after = await page.evaluate("() => window.scrollY")
    if after != before:
        return f"scrolled {direction} {abs(after - before)}px"
    # Window didn't move: many SPAs scroll an inner container. Send a wheel event at the center.
    vw = page.viewport_size["width"] if page.viewport_size else 1280
    await page.mouse.move(vw / 2, vh / 2)
    await page.mouse.wheel(0, dy)
    return f"scrolled {direction} (mouse wheel over the page center)"


def _short_reason(e: Exception) -> str:
    msg = str(e).splitlines()[0] if str(e) else type(e).__name__
    return msg[:200]

"""Set-of-marks screenshots: the viewport with a numbered box drawn on every interactive element.

The numbers match the element list, so the model can connect what it sees (layout, icons,
colours, images) with what it can act on. Drawing happens in Pillow, not in the page, so
the DOM the site sees is never modified.
"""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import Page

from browserai.perception.state import PageState

# High-contrast palette; consecutive indices get different colours so nearby boxes are distinguishable.
_COLORS = [
    (230, 25, 75), (0, 130, 200), (60, 180, 75), (245, 130, 48), (145, 30, 180),
    (0, 128, 128), (240, 50, 230), (128, 0, 0), (0, 0, 128), (170, 110, 40),
]

_font_cache: dict[int, Any] = {}


def _font(size: int):
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.load_default(size=size)
        except TypeError:  # Pillow without FreeType support
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


Rect = tuple[int, int, int, int]


def _overlap(a: Rect, b: Rect) -> int:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0


def _place_label(box: Rect, tw: int, th: int, size: tuple[int, int],
                 placed: list[Rect], others: list[Rect]) -> Rect:
    """Pick the tag position that hides the least: other tags (heavily) and other elements."""
    x0, y0, x1, y1 = box
    width, height = size
    candidates = [
        (x0 - tw, y0),          # left of the box
        (x0, y0 - th),          # above, left-aligned
        (x1 - tw, y0 - th),     # above, right-aligned
        (x1, y0),               # right of the box
        (x0, y1),               # below
        (x0, y0),               # inside the top-left corner (last resort)
    ]
    best, best_cost = None, None
    for cx, cy in candidates:
        cx = min(max(cx, 0), width - tw)
        cy = min(max(cy, 0), height - th)
        r = (cx, cy, cx + tw, cy + th)
        cost = sum(_overlap(r, p) for p in placed) * 10 + sum(_overlap(r, o) for o in others)
        if best_cost is None or cost < best_cost:
            best, best_cost = r, cost
            if cost == 0:
                break
    return best  # type: ignore[return-value]


def annotate(png: bytes, state: PageState, *, quality: int = 70) -> bytes:
    """Draw numbered boxes for in-view, uncovered elements; return a JPEG."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = _font(12)
    width, height = img.size

    items: list[tuple[int, Rect]] = []
    for el in state.elements:
        if not el.in_viewport or el.occluded:
            continue
        b = el.box
        r = (max(b.x, 0), max(b.y, 0), min(b.x + b.w, width - 1), min(b.y + b.h, height - 1))
        if r[2] > r[0] and r[3] > r[1]:
            items.append((el.index, r))

    for index, r in items:  # boxes first, so tags are drawn on top of every outline
        draw.rectangle(r, outline=_COLORS[index % len(_COLORS)] + (255,), width=2)

    boxes = [r for _, r in items]
    placed: list[Rect] = []
    for i, (index, r) in enumerate(items):
        color = _COLORS[index % len(_COLORS)]
        label = str(index)
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), label, font=font)
        tw, th = tx1 - tx0 + 4, ty1 - ty0 + 3
        others = boxes[:i] + boxes[i + 1:]
        lr = _place_label(r, tw, th, (width, height), placed, others)
        placed.append(lr)
        draw.rectangle(lr, fill=color + (235,))
        draw.text((lr[0] + 2 - tx0, lr[1] + 1 - ty0), label, fill=(255, 255, 255), font=font)

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


async def annotated_screenshot(page: Page, state: PageState) -> bytes:
    png = await page.screenshot(type="png", full_page=False, animations="disabled")
    return annotate(png, state)


def image_block(jpeg: bytes) -> dict[str, Any]:
    """Messages API image content block."""
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()},
    }

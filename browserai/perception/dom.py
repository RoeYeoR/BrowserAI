"""Runs extract.js in every frame and merges the results into one indexed PageState.

Responsibilities on the Python side:
  * run the extractor in all frames concurrently (asyncio.gather)
  * translate iframe-relative coordinates into page coordinates
  * hand out element indices that stay the same across snapshots of the same document
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Frame, Page

from browserai.perception.state import Box, Element, PageState

EXTRACT_JS = (Path(__file__).parent / "extract.js").read_text(encoding="utf-8")

_ELEMENT_FIELDS = {
    "kind", "tag", "name", "occluded", "via_label", "in_shadow", "disabled", "checked",
    "expanded", "required", "value", "placeholder", "href", "options", "options_total", "new_tab",
    "popup", "invalid", "error", "sensitive",
}


class DomExtractor:
    """Stateful: remembers which (document, element) got which index, so `[12]` keeps
    pointing at the same element while the page is re-rendered around it."""

    def __init__(self, max_elements: int = 400) -> None:
        self.max_elements = max_elements
        self._index: dict[tuple[str, int], int] = {}
        self._next = 1
        self._main_doc: str | None = None
        self._last: list[tuple[tuple[str, int], str, str, int]] = []  # previous snapshot's elements

    async def snapshot(self, page: Page) -> PageState:
        frames = [f for f in page.frames if not f.is_detached()]
        results = await asyncio.gather(
            *(self._extract_frame(f, page.main_frame) for f in frames),
            return_exceptions=True,
        )

        main_raw: dict[str, Any] | None = None
        notes: list[str] = []
        per_frame: list[tuple[Frame, dict[str, Any], tuple[float, float, Box | None]]] = []
        for frame, res in zip(frames, results):
            if isinstance(res, BaseException):
                if frame is page.main_frame:
                    raise res
                notes.append(f"could not read iframe {frame.url[:80]}: {type(res).__name__}")
                continue
            if res is None:  # hidden / zero-size iframe
                continue
            raw, offset = res
            if frame is page.main_frame:
                main_raw = raw
            per_frame.append((frame, raw, offset))

        assert main_raw is not None
        # New main document (navigation or reload): restart numbering from 1.
        if main_raw["doc_id"] != self._main_doc:
            self._main_doc = main_raw["doc_id"]
            self._index.clear()
            self._last = []
            self._next = 1

        vw, vh = main_raw["viewport"]["w"], main_raw["viewport"]["h"]
        elements: list[Element] = []
        keys: list[tuple[str, int]] = []
        truncated = False
        for frame, raw, (ox, oy, clip) in per_frame:
            truncated |= raw["truncated"]
            is_main = frame is page.main_frame
            for item in raw["elements"]:
                b = item["box"]
                box = Box(int(b["x"] + ox), int(b["y"] + oy), b["w"], b["h"])
                cx, cy = box.center
                in_vp = item["in_viewport"] and 0 <= cx < vw and 0 <= cy < vh
                if clip is not None:  # inside an iframe: must also be inside the iframe's box
                    in_vp = in_vp and clip.x <= cx < clip.x + clip.w and clip.y <= cy < clip.y + clip.h
                keys.append((raw["doc_id"], item["bai_id"]))
                elements.append(
                    Element(
                        index=0,  # assigned below
                        bai_id=item["bai_id"],
                        box=box,
                        in_viewport=in_vp,
                        frame_url=None if is_main else frame.url,
                        frame=frame,
                        **{k: v for k, v in item.items() if k in _ELEMENT_FIELDS},
                    )
                )

        self._assign_indices(elements, keys)
        return PageState(
            url=main_raw["url"],
            title=main_raw["title"],
            viewport=(vw, vh),
            scroll=(main_raw["scroll"]["x"], main_raw["scroll"]["y"]),
            page_size=(main_raw["page"]["w"], main_raw["page"]["h"]),
            elements=elements,
            truncated=truncated,
            notes=notes,
            doc_id=main_raw["doc_id"],
            text=main_raw.get("text", ""),
        )

    async def _extract_frame(
        self, frame: Frame, main: Frame
    ) -> tuple[dict[str, Any], tuple[float, float, Box | None]] | None:
        offset: tuple[float, float, Box | None] = (0.0, 0.0, None)
        if frame is not main:
            # bounding_box() is relative to the main frame's viewport, even for nested iframes.
            handle = await frame.frame_element()
            bb = await handle.bounding_box()
            if not bb or bb["width"] < 1 or bb["height"] < 1:
                return None
            clip = Box(int(bb["x"]), int(bb["y"]), int(bb["width"]), int(bb["height"]))
            offset = (bb["x"], bb["y"], clip)
        try:
            raw = await frame.evaluate(EXTRACT_JS, {"maxElements": self.max_elements})
        except PlaywrightError:
            if frame is main:
                raise
            return None  # iframe navigated / detached mid-snapshot
        return raw, offset

    def _assign_indices(self, elements: list[Element], keys: list[tuple[str, int]]) -> None:
        """Known nodes keep their index. A NEW node that replaces a node which vanished since
        the last snapshot - same document, kind and name, and the match is unambiguous - inherits
        the vanished node's index: that is a framework re-render, not a new element."""
        present = set(keys)
        fresh: dict[tuple, list[tuple[Element, tuple[str, int]]]] = defaultdict(list)
        for el, key in zip(elements, keys):
            if key in self._index:
                el.index = self._index[key]
            elif el.name:
                fresh[(key[0], el.kind, el.name)].append((el, key))

        vanished: dict[tuple, list[int]] = defaultdict(list)
        for key, kind, name, index in self._last:
            if key not in present and name:
                vanished[(key[0], kind, name)].append(index)

        for sig, new_items in fresh.items():
            old = vanished.get(sig)
            if old and len(old) == 1 and len(new_items) == 1:
                el, key = new_items[0]
                el.index = self._index[key] = old[0]

        for el, key in zip(elements, keys):
            if key not in self._index:
                el.index = self._index_for(*key)
        self._last = [(key, el.kind, el.name, el.index) for el, key in zip(elements, keys)]

    def _index_for(self, doc_id: str, bai_id: int) -> int:
        key = (doc_id, bai_id)
        idx = self._index.get(key)
        if idx is None:
            idx = self._index[key] = self._next
            self._next += 1
        return idx

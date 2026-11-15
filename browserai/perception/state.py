"""PageState: what the agent "sees" at one step - indexed interactive elements plus page metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _compact_href(href: str, limit: int = 70) -> str:
    """Drop long query strings (usually tracking params) - they cost tokens every step."""
    href = href.rstrip("…")
    if len(href) <= limit:
        return href
    base, _, query = href.partition("?")
    if query:
        return (base[:limit] + "?…") if len(base) <= limit else base[:limit] + "…"
    return href[:limit] + "…"


def _fmt(v: object) -> str:
    if isinstance(v, str):
        return json.dumps(v[:40] + ("…" if len(v) > 40 else ""), ensure_ascii=False)
    return str(v)


def _short_list(elements: list[Element], limit: int) -> str:
    items = [f"[{e.index}] {e.kind} {_fmt(e.name)}" if e.name else f"[{e.index}] {e.kind}" for e in elements[:limit]]
    return ", ".join(items) + (f" …(+{len(elements) - limit})" if len(elements) > limit else "")


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass
class Element:
    """One interactive element. `index` is the stable number shown to the model."""

    index: int
    bai_id: int
    kind: str
    tag: str
    name: str
    box: Box  # page (main-frame viewport) coordinates
    in_viewport: bool
    occluded: bool = False
    via_label: bool = False
    in_shadow: bool = False
    frame_url: str | None = None  # set only for elements inside an iframe
    disabled: bool = False
    checked: bool | None = None
    expanded: bool | None = None
    required: bool = False
    value: str | None = None
    placeholder: str | None = None
    href: str | None = None
    options: list[str] | None = None
    options_total: int | None = None
    new_tab: bool = False  # link with target=_blank
    popup: str | None = None  # aria-haspopup: "listbox", "menu", "dialog", ...
    invalid: bool = False  # failed validation (aria-invalid or native constraint after interaction)
    error: str | None = None  # validation message, when available
    sensitive: str | None = None  # "password" | "payment" | "otp": never filled by the agent
    # Playwright Frame that owns the element; used to build a locator. Not serialized.
    frame: Any = field(default=None, repr=False, compare=False)
    refound: bool = field(default=False, repr=False, compare=False)  # re-located after a re-render

    @property
    def selector(self) -> str:
        return f'[data-bai-id="{self.bai_id}"]'

    def describe(self) -> str:
        """Compact one-line description for the model, e.g. `[3] button "Search"`."""
        parts = [f"[{self.index}] {self.kind}"]
        if self.name:
            parts.append(json.dumps(self.name, ensure_ascii=False))
        if self.checked is not None:
            parts.append("☑" if self.checked else "☐")
        if self.value is not None and self.kind not in ("checkbox", "radio"):
            label = "selected" if self.options is not None else "value"
            parts.append(f"{label}={json.dumps(self.value, ensure_ascii=False)}")
        if self.placeholder and self.placeholder != self.name:
            parts.append(f"placeholder={json.dumps(self.placeholder, ensure_ascii=False)}")
        if self.options is not None:
            more = f" …(+{self.options_total - len(self.options)})" if self.options_total else ""
            parts.append("options: " + " | ".join(self.options) + more)
        if self.href:
            parts.append(f"→ {_compact_href(self.href)}")
        flags = []
        if self.disabled:
            flags.append("disabled")
        if self.required:
            flags.append("required")
        if self.expanded is not None:
            flags.append("expanded" if self.expanded else "collapsed")
        if self.sensitive:
            flags.append(f"{self.sensitive} field - user only")
        if self.popup:
            flags.append(f"opens a {self.popup}")
        if self.invalid:
            flags.append(f"INVALID: {self.error}" if self.error else "INVALID")
        if self.new_tab:
            flags.append("opens in a new tab")
        if self.occluded:
            flags.append("covered by another element")
        if self.frame_url:
            flags.append("in iframe")
        if flags:
            parts.append("(" + ", ".join(flags) + ")")
        return " ".join(parts)


@dataclass
class PageState:
    url: str
    title: str
    viewport: tuple[int, int]
    scroll: tuple[int, int]
    page_size: tuple[int, int]
    elements: list[Element]
    truncated: bool = False
    notes: list[str] = field(default_factory=list)
    doc_id: str = ""  # changes when the main document is replaced (navigation / reload)
    text: str = field(default="", repr=False)  # visible page text, for change detection only

    def get(self, index: int) -> Element | None:
        for el in self.elements:
            if el.index == index:
                return el
        return None

    def to_prompt(self, max_offscreen: int = 60) -> str:
        """Render the state as text for the model: on-screen elements first, then off-screen ones."""
        vw, vh = self.viewport
        sx, sy = self.scroll
        pw, ph = self.page_size
        on = [e for e in self.elements if e.in_viewport]
        above = [e for e in self.elements if not e.in_viewport and e.box.y + e.box.h <= 0]
        below = [e for e in self.elements if not e.in_viewport and e not in above]

        lines = [
            f"URL: {self.url}",
            f"Title: {self.title}",
            f"Viewport {vw}x{vh}, scrolled to y={sy} of {ph}px page height"
            + (f", x={sx}" if sx else ""),
            "",
            f"Interactive elements in view ({len(on)}):",
        ]
        lines += [e.describe() for e in on] or ["(none)"]

        for label, group in (("Above the viewport (scroll up)", above), ("Below the viewport (scroll down)", below)):
            if not group:
                continue
            lines.append("")
            lines.append(f"{label} ({len(group)}):")
            lines += [e.describe() for e in group[:max_offscreen]]
            if len(group) > max_offscreen:
                lines.append(f"… {len(group) - max_offscreen} more")

        if self.truncated:
            lines.append("\n(note: element list truncated - page has more interactive elements)")
        for note in self.notes:
            lines.append(f"(note: {note})")
        return "\n".join(lines)

    def diff(self, before: PageState | None, max_items: int = 6) -> list[str]:
        """Human-readable list of what changed since `before`. Empty list = nothing changed."""
        if before is None:
            return []
        changes: list[str] = []
        if self.url != before.url:
            changes.append(f"URL changed: {before.url} → {self.url}")
        elif self.doc_id != before.doc_id:
            changes.append("the page reloaded")
        if self.title != before.title:
            changes.append(f"title changed to {self.title!r}")
        if changes and self.doc_id != before.doc_id:
            return changes  # new document: element numbers restarted, element-level diff is meaningless

        old = {e.index: e for e in before.elements}
        new = {e.index: e for e in self.elements}
        appeared = [e for i, e in new.items() if i not in old]
        gone = [e for i, e in old.items() if i not in new]
        if appeared:
            changes.append(f"{len(appeared)} new element(s): " + _short_list(appeared, max_items))
        if gone:
            changes.append(f"{len(gone)} element(s) disappeared: " + _short_list(gone, max_items))

        updates = []
        for i in sorted(old.keys() & new.keys()):
            a, b = old[i], new[i]
            for attr, label in (("name", "text"), ("value", "value"), ("checked", "checked"),
                                ("expanded", "expanded"), ("disabled", "disabled"),
                                ("invalid", "invalid")):
                va, vb = getattr(a, attr), getattr(b, attr)
                if va != vb:
                    updates.append(f"[{i}] {label}: {_fmt(va)} → {_fmt(vb)}")
            if a.occluded != b.occluded:
                updates.append(f"[{i}] {'is now covered' if b.occluded else 'is no longer covered'}")
        if updates:
            more = f" …(+{len(updates) - max_items})" if len(updates) > max_items else ""
            changes.append("changed: " + "; ".join(updates[:max_items]) + more)

        # Plain-text changes (status messages, results, errors) that aren't interactive elements.
        old_lines = {ln.strip() for ln in before.text.splitlines() if ln.strip()}
        new_lines = [ln.strip() for ln in self.text.splitlines() if ln.strip()]
        added = list(dict.fromkeys(ln for ln in new_lines if ln not in old_lines))
        removed = len(old_lines - set(new_lines))
        if added:
            shown = "; ".join(_fmt(ln[:80]) for ln in added[:3])
            more = f" …(+{len(added) - 3} lines)" if len(added) > 3 else ""
            changes.append(f"new text on the page: {shown}{more}")
        elif removed:
            changes.append(f"{removed} line(s) of text disappeared from the page")

        if self.scroll != before.scroll:
            changes.append(f"scrolled to y={self.scroll[1]} (was {before.scroll[1]})")
        return changes

    def fingerprint(self) -> str:
        """Hash of what matters for "did the page change?" checks (used by the verify step in M4)."""
        data = [self.url, self.scroll, hashlib.sha1(self.text.encode()).hexdigest()] + [
            (e.index, e.kind, e.name, e.value, e.checked, e.expanded, e.disabled, e.in_viewport)
            for e in self.elements
        ]
        return hashlib.sha1(json.dumps(data, default=str).encode()).hexdigest()[:16]

"""Safety layer between the model and the browser.

Three kinds of rules, checked before every tool call:

* hard blocks - never performed, whatever the model says: typing into password, payment
  or one-time-code fields.
* approval required - actions with real-world effects (submitting, buying, sending,
  deleting, uploading a file, accepting a confirmation dialog). A human approves each one;
  in non-interactive runs they are refused unless the run was started with approval mode
  "allow".
* domain policy - navigation outside the allowed domains is blocked (enforced at the
  network layer by BrowserSession, so clicks and redirects are covered too).

The model is told why an action was refused, so it can explain or choose another route.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

from browserai.perception.state import Element

ApprovalMode = Literal["ask", "allow", "deny"]

# Button/link text that usually means an irreversible or outward-facing action.
RISKY_TEXT = re.compile(
    r"\b("
    r"buy|purchase|pay|place (an |my )?order|order now|check ?out|"
    r"submit|send|post|publish|share|"
    r"delete|remove|erase|destroy|"
    r"transfer|withdraw|donate|"
    r"subscribe|unsubscribe|sign up|register|"
    r"book|reserve|"
    r"cancel (my |the )?(subscription|account|order|booking)|close (my )?account|deactivate"
    r")\b",
    re.IGNORECASE,
)

SENSITIVE_REASONS = {
    "password": "it is a password field - the agent never enters passwords; the user must log in themselves",
    "payment": "it is a payment field - the agent never enters card or bank details",
    "otp": "it is a one-time-code field - verification codes must be entered by the user",
}


@dataclass
class Review:
    blocked: str | None = None  # hard block reason
    needs_approval: str | None = None  # human-readable description of the risky action


def review(tool: str, args: dict[str, Any], el: Element | None, form_submit_text: str = "") -> Review:
    """Classify a validated tool call. Pure function: easy to test and to reason about."""
    if tool == "type_text" and el is not None and el.sensitive:
        return Review(blocked=f"Not typed into [{el.index}]: {SENSITIVE_REASONS.get(el.sensitive, el.sensitive)}.")

    if tool == "click" and el is not None and el.name and RISKY_TEXT.search(el.name):
        return Review(needs_approval=f'click [{el.index}] {el.kind} "{el.name}"')

    if tool == "type_text" and args.get("press_enter") and form_submit_text and RISKY_TEXT.search(form_submit_text):
        return Review(needs_approval=f'press Enter in [{args["index"]}], submitting a form ("{form_submit_text}")')

    if tool == "upload_file" and el is not None:
        return Review(needs_approval=f'upload "{args.get("filename")}" to this site via [{el.index}]')

    if tool == "handle_next_dialog" and args.get("accept"):
        return Review(needs_approval="accept the next confirmation dialog")

    return Review()


def domain_allowed(url: str, allowed: list[str]) -> bool:
    """True if `url`'s host is one of `allowed` or a subdomain of one. Empty list = everything."""
    if not allowed:
        return True
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return True  # about:, data:, file: etc. are not network navigations
    host = (parsed.hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in (a.lower().strip(".") for a in allowed))


Approver = Callable[[str], Awaitable[bool]]


@dataclass
class SafetyGuard:
    mode: ApprovalMode = "deny"
    approver: Approver | None = None  # used in "ask" mode (e.g. a CLI prompt)
    log: list[dict[str, Any]] = field(default_factory=list)

    async def approve(self, description: str) -> bool:
        if self.mode == "allow":
            ok = True
        elif self.mode == "ask" and self.approver is not None:
            try:
                ok = await self.approver(description)
            except (EOFError, KeyboardInterrupt, OSError):
                # Nobody can answer (no usable terminal): treat as "no" and stop asking.
                ok = False
                self.mode = "deny"
        else:
            ok = False
        self.log.append({"action": description, "approved": ok, "mode": self.mode})
        return ok

    def refusal(self, description: str) -> str:
        if self.mode == "ask":
            return (f"Not performed: the user declined to {description}. Do not retry it; "
                    "continue another way or finish with `done` and explain.")
        return (f"Not performed: '{description}' needs the user's approval, and approvals are "
                "disabled for this run. Stop before this step and finish with `done`, telling the "
                "user exactly what is left for them to do.")

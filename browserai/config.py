"""Runtime configuration, loaded from environment variables and an optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("BROWSERAI_MODEL", "claude-opus-5-5")
    effort: str = os.getenv("BROWSERAI_EFFORT", "medium")
    headless: bool = _bool("BROWSERAI_HEADLESS", False)
    max_steps: int = int(os.getenv("BROWSERAI_MAX_STEPS", "30"))
    vision: bool = _bool("BROWSERAI_VISION", True)  # send annotated screenshots to the model
    # Comma-separated in BROWSERAI_ALLOWED_DOMAINS; empty = any domain. Subdomains are included.
    allowed_domains: tuple[str, ...] = tuple(
        d.strip() for d in os.getenv("BROWSERAI_ALLOWED_DOMAINS", "").split(",") if d.strip()
    )
    approval: str = os.getenv("BROWSERAI_APPROVAL", "ask")  # ask | allow | deny
    max_cost_usd: float = float(os.getenv("BROWSERAI_MAX_COST_USD", "2.0"))
    upload_dir: str = os.getenv("BROWSERAI_UPLOAD_DIR", "uploads")  # the only folder upload_file may read
    trace_dir: str | None = os.getenv("BROWSERAI_TRACE_DIR", "runs")  # None/"" disables tracing
    viewport_width: int = 1280
    viewport_height: int = 800
    nav_timeout_ms: int = 30_000
    settle_timeout_s: float = float(os.getenv("BROWSERAI_SETTLE_TIMEOUT", "6"))


def get_settings(**overrides: object) -> Settings:
    """Return settings, with any non-None keyword overrides applied (e.g. from CLI flags)."""
    base = Settings()
    clean = {k: v for k, v in overrides.items() if v is not None}
    return Settings(**{**base.__dict__, **clean}) if clean else base

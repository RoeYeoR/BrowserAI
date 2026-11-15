"""Thin async wrapper around the Anthropic Messages API.

All requests share the same model, effort and refusal-fallback settings. Extra request
features (context editing, thinking display) are passed through by the agent loop.
"""

from __future__ import annotations

from typing import Any

import anthropic

from browserai.config import Settings

# Server-side refusal fallback: if the model declines, the API reroutes the request
# to a suitable fallback model instead of returning an empty refusal.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = anthropic.AsyncAnthropic()

    async def create(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 16000,
        betas: list[str] | None = None,
        **extra: Any,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "output_config": {"effort": self.settings.effort},
            "betas": [_FALLBACK_BETA, *(betas or [])],
            "fallbacks": "default",
            **extra,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        try:
            response = await self._client.beta.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise LLMError("Anthropic authentication failed - check ANTHROPIC_API_KEY in .env") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Rate limited by the Anthropic API - retry shortly") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Could not reach the Anthropic API - check your connection") from e
        except TypeError as e:
            # The SDK raises TypeError at request time when no credential source is found.
            if "authentication" not in str(e):
                raise
            raise LLMError(
                "No Anthropic credentials found. Copy .env.example to .env and set "
                "ANTHROPIC_API_KEY (or run with --no-llm to test only the browser)."
            ) from e

        if response.stop_reason == "refusal":
            raise LLMError("The model declined this request")
        return response

    async def ask(self, prompt: str, *, system: str | None = None, max_tokens: int = 2000) -> str:
        """Single-turn text question; returns the concatenated text blocks."""
        response = await self.create(
            [{"role": "user", "content": prompt}], system=system, max_tokens=max_tokens
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()

    async def aclose(self) -> None:
        await self._client.close()

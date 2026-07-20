from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from codeweave.models import Message, ProviderConfig, StreamEvent


class ProviderError(RuntimeError):
    """Base error for provider failures."""


class LLMProvider(Protocol):
    mode: str

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]: ...


@dataclass(slots=True)
class ProviderBase:
    config: ProviderConfig

    @property
    def mode(self) -> str:
        return self.config.mode


def require_api_key(config: ProviderConfig) -> str:
    if not config.api_key:
        raise ProviderError(f"{config.mode} provider requires an API key")
    return config.api_key

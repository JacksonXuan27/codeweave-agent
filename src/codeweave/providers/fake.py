from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any

from codeweave.models import Message, ProviderConfig, StreamEnd, StreamEvent, TextDelta
from codeweave.providers.base import ProviderBase


class FakeProvider(ProviderBase):
    """Deterministic provider used by tests and offline demos."""

    def __init__(self, events: Iterable[StreamEvent | str] | None = None) -> None:
        super().__init__(ProviderConfig(mode="fake", model="fake"))
        self.events = list(events or [StreamEnd()])
        self.requests: list[list[Message]] = []

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del tools
        self.requests.append(list(messages))
        for event in self.events:
            yield TextDelta(event) if isinstance(event, str) else event

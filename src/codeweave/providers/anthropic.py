from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from codeweave.models import Message, ProviderConfig, StreamEnd, StreamEvent, TextDelta, ToolCall
from codeweave.providers.base import ProviderBase, ProviderError, require_api_key


class AnthropicProvider(ProviderBase):
    """Adapter for Anthropic's messages streaming API."""

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ProviderError("anthropic package is not installed") from exc

        system_messages = [message.content for message in messages if str(message.role) == "system"]
        request_messages = [
            message.to_dict() for message in messages if str(message.role) != "system"
        ]
        client = AsyncAnthropic(api_key=require_api_key(self.config), base_url=self.config.base_url)
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": request_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": True,
        }
        if system_messages:
            request["system"] = "\n\n".join(system_messages)
        if tools:
            request["tools"] = list(tools)

        try:
            stream = await client.messages.create(**request)
            async for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "content_block_delta":
                    text = getattr(getattr(event, "delta", None), "text", None)
                    if text:
                        yield TextDelta(text)
                elif event_type == "message_delta":
                    stop_reason = getattr(getattr(event, "delta", None), "stop_reason", None)
                    if stop_reason:
                        yield StreamEnd(stop_reason=stop_reason)
            yield StreamEnd(stop_reason="stop")
        except Exception as exc:
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(f"anthropic request failed: {exc}") from exc

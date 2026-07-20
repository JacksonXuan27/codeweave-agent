from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from codeweave.models import Message, ProviderConfig, StreamEnd, StreamEvent, TextDelta, ToolCallComplete
from codeweave.providers.base import ProviderBase, ProviderError, require_api_key


class OpenAIProvider(ProviderBase):
    """Adapter for the official OpenAI API."""

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ProviderError("openai package is not installed") from exc

        client = AsyncOpenAI(api_key=require_api_key(self.config), base_url=self.config.base_url)
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.to_dict() for message in messages],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": True,
        }
        if tools:
            request["tools"] = list(tools)

        try:
            response = await client.chat.completions.create(**request)
            tool_buffers: dict[int, dict[str, str]] = {}
            async for chunk in response:
                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue
                delta = choice.delta
                if delta.content:
                    yield TextDelta(delta.content)
                for tool_delta in delta.tool_calls or []:
                    buffer = tool_buffers.setdefault(tool_delta.index, {"id": "", "name": "", "arguments": ""})
                    if tool_delta.id:
                        buffer["id"] = tool_delta.id
                    if tool_delta.function:
                        if tool_delta.function.name:
                            buffer["name"] = tool_delta.function.name
                        if tool_delta.function.arguments:
                            buffer["arguments"] += tool_delta.function.arguments
                if choice.finish_reason == "tool_calls":
                    for buffer in tool_buffers.values():
                        try:
                            arguments = json.loads(buffer["arguments"] or "{}")
                        except json.JSONDecodeError as exc:
                            raise ProviderError("provider returned invalid tool arguments") from exc
                        yield ToolCallComplete(
                            __import__("codeweave.models", fromlist=["ToolCall"]).ToolCall(
                                buffer["id"], buffer["name"], arguments
                            )
                        )
            yield StreamEnd(stop_reason="stop")
        except Exception as exc:
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(f"openai request failed: {exc}") from exc

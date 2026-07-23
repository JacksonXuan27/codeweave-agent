from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass

from codeweave.models import (
    Message,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallComplete,
    ToolCallDelta,
    ToolResult,
)
from codeweave.permissions import PermissionChecker, PermissionMode
from codeweave.prompts import PromptComposer
from codeweave.providers import LLMProvider
from codeweave.tools import ToolRegistry


@dataclass(slots=True)
class _ToolCallBuffer:
    name: str
    arguments: str = ""


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        *,
        max_turns: int = 20,
        prompt_composer: PromptComposer | None = None,
        permission_checker: PermissionChecker | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.provider = provider
        self.tools = tools
        self.max_turns = max_turns
        self.prompt_composer = prompt_composer
        self.permission_checker = permission_checker or PermissionChecker(PermissionMode.CONFIRM)
        self.messages: list[Message] = []
        self.turn_count = 0

    async def run(
        self,
        prompt: str,
        *,
        runtime_context: Mapping[str, object] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        messages = [Message.user(prompt)]
        if self.prompt_composer is not None:
            tool_descriptions = [
                f"{spec.name}: {spec.description}" for spec in self.tools.specs()
            ]
            system_prompt = self.prompt_composer.compose(
                tool_descriptions=tool_descriptions,
                task=prompt,
                runtime_context=runtime_context,
            )
            messages.insert(0, Message.system(system_prompt))
        elif runtime_context:
            raise ValueError("runtime_context requires a prompt_composer")

        self.messages = messages
        self.turn_count = 0

        for turn_index in range(self.max_turns):
            self.turn_count = turn_index + 1
            content_parts: list[str] = []
            call_order: list[str] = []
            completed_calls: dict[str, ToolCall] = {}
            call_buffers: dict[str, _ToolCallBuffer] = {}

            try:
                async for event in self.provider.stream(
                    self.messages,
                    self.tools.schemas() or None,
                ):
                    if isinstance(event, TextDelta):
                        content_parts.append(event.text)
                    elif isinstance(event, ToolCallComplete):
                        call_id = event.tool_call.id
                        if call_id not in call_order:
                            call_order.append(call_id)
                        completed_calls[call_id] = event.tool_call
                        call_buffers.pop(call_id, None)
                    elif isinstance(event, ToolCallDelta):
                        call_id = event.tool_call_id
                        if call_id not in call_order:
                            call_order.append(call_id)
                        buffer = call_buffers.setdefault(
                            call_id,
                            _ToolCallBuffer(name=event.name),
                        )
                        if event.name:
                            buffer.name = event.name
                        buffer.arguments += event.arguments_fragment
                    yield event
            except Exception as exc:
                yield TextDelta(f"Agent error: {exc}")
                yield StreamEnd(stop_reason="error")
                return

            content = "".join(content_parts)
            invalid_calls: dict[str, str] = {}
            for call_id, buffer in call_buffers.items():
                try:
                    arguments = json.loads(buffer.arguments or "{}")
                except json.JSONDecodeError:
                    invalid_calls[call_id] = "invalid JSON arguments"
                    completed_calls[call_id] = ToolCall(call_id, buffer.name, {})
                    continue
                if not isinstance(arguments, dict):
                    invalid_calls[call_id] = "tool arguments must be a JSON object"
                    completed_calls[call_id] = ToolCall(call_id, buffer.name, {})
                    continue
                completed_calls[call_id] = ToolCall(call_id, buffer.name, arguments)

            if not call_order:
                if content:
                    self.messages.append(Message.assistant(content))
                return

            ordered_calls = [
                completed_calls[call_id]
                for call_id in call_order
                if call_id in completed_calls
            ]
            self.messages.append(Message.assistant(content, ordered_calls))

            for call_id in call_order:
                if call_id in invalid_calls:
                    result = ToolResult(call_id, False, error=invalid_calls[call_id])
                else:
                    tool_call = completed_calls[call_id]
                    spec = self.tools.get(tool_call.name)
                    if spec is None:
                        result = self.tools.execute(tool_call)
                    else:
                        decision = self.permission_checker.check(spec, tool_call)
                        result = (
                            self.tools.execute(tool_call)
                            if decision.allowed
                            else ToolResult(tool_call.id, False, error=decision.reason)
                        )
                self.messages.append(result.to_message())

            if self.turn_count >= self.max_turns:
                yield TextDelta("Agent stopped after reaching the turn limit.")
                yield StreamEnd(stop_reason="max_turns")
                return

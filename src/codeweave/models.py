from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class Message:
    role: Role | str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(Role.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(Role.USER, content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(Role.ASSISTANT, content, tool_calls or [])

    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> Message:
        return cls(Role.TOOL, content, tool_call_id=tool_call_id)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": str(self.role), "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str
    ok: bool
    output: str = ""
    error: str = ""

    def to_message(self) -> Message:
        content = self.output if self.ok else f"Tool error: {self.error}"
        return Message.tool(content, self.tool_call_id)


@dataclass(slots=True)
class TextDelta:
    text: str


@dataclass(slots=True)
class ToolCallDelta:
    tool_call_id: str
    name: str
    arguments_fragment: str


@dataclass(slots=True)
class ToolCallComplete:
    tool_call: ToolCall


@dataclass(slots=True)
class StreamEnd:
    stop_reason: str = "stop"
    input_tokens: int = 0
    output_tokens: int = 0


StreamEvent = TextDelta | ToolCallDelta | ToolCallComplete | StreamEnd


@dataclass(slots=True)
class ProviderConfig:
    mode: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0


def messages_to_dicts(messages: list[Message]) -> list[Mapping[str, Any]]:
    return [message.to_dict() for message in messages]

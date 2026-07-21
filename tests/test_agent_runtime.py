from codeweave import Agent
from codeweave.models import (
    StreamEnd,
    TextDelta,
    ToolCall,
    ToolCallComplete,
    ToolCallDelta,
)
from codeweave.providers import FakeProvider
from codeweave.tools import ToolRegistry, ToolSpec


async def test_agent_runs_tool_calls_until_provider_stops() -> None:
    provider = FakeProvider(
        [
            ToolCallComplete(ToolCall("call-1", "echo", {"value": "ok"})),
            StreamEnd(stop_reason="tool_calls"),
            TextDelta("done"),
            StreamEnd(stop_reason="stop"),
        ]
    )
    tools = ToolRegistry()
    tools.register(
        ToolSpec(
            "echo",
            "Echo a value",
            lambda arguments: arguments["value"],
            {"type": "object"},
        )
    )
    agent = Agent(provider, tools, max_turns=3)

    events = [event async for event in agent.run("use echo")]

    assert [event.text for event in events if isinstance(event, TextDelta)] == ["done"]
    assert len(provider.requests) == 2
    assert provider.requests[1][-1].role == "tool"


async def test_agent_stops_after_turn_limit() -> None:
    provider = FakeProvider([ToolCallComplete(ToolCall("call-1", "echo", {}))])
    tools = ToolRegistry()
    tools.register(ToolSpec("echo", "Echo", lambda arguments: "x", {"type": "object"}))
    agent = Agent(provider, tools, max_turns=1)

    events = [event async for event in agent.run("loop")]

    assert any("turn limit" in event.text.lower() for event in events if isinstance(event, TextDelta))


async def test_agent_executes_multiple_tool_calls_in_order() -> None:
    provider = FakeProvider(
        [
            ToolCallComplete(ToolCall("call-1", "first", {})),
            ToolCallComplete(ToolCall("call-2", "second", {})),
            StreamEnd(stop_reason="tool_calls"),
            TextDelta("done"),
            StreamEnd(stop_reason="stop"),
        ]
    )
    tools = ToolRegistry()
    calls: list[str] = []
    tools.register(
        ToolSpec(
            "first",
            "First tool",
            lambda arguments: calls.append("first") or "one",
            {"type": "object"},
        )
    )
    tools.register(
        ToolSpec(
            "second",
            "Second tool",
            lambda arguments: calls.append("second") or "two",
            {"type": "object"},
        )
    )
    agent = Agent(provider, tools, max_turns=2)

    events = [event async for event in agent.run("run both")]

    assert [event.text for event in events if isinstance(event, TextDelta)] == ["done"]
    assert calls == ["first", "second"]
    assert [message.content for message in provider.requests[1][-2:]] == ["one", "two"]


async def test_agent_returns_tool_errors_for_provider_recovery() -> None:
    provider = FakeProvider(
        [
            ToolCallComplete(ToolCall("call-1", "broken", {})),
            StreamEnd(stop_reason="tool_calls"),
            TextDelta("recovered"),
            StreamEnd(stop_reason="stop"),
        ]
    )
    tools = ToolRegistry()

    def broken_tool(arguments: dict[str, object]) -> str:
        raise RuntimeError("boom")

    tools.register(ToolSpec("broken", "Broken tool", broken_tool, {"type": "object"}))
    agent = Agent(provider, tools, max_turns=2)

    events = [event async for event in agent.run("recover")]

    assert [event.text for event in events if isinstance(event, TextDelta)] == ["recovered"]
    assert "Tool error: boom" in provider.requests[1][-1].content


async def test_agent_surfaces_provider_errors() -> None:
    class FailingProvider:
        mode = "fake"

        async def stream(self, messages, tools=None):
            raise RuntimeError("offline")
            yield TextDelta("")

    agent = Agent(FailingProvider(), ToolRegistry())

    events = [event async for event in agent.run("fail safely")]

    assert isinstance(events[0], TextDelta)
    assert events[0].text == "Agent error: offline"
    assert isinstance(events[1], StreamEnd)
    assert events[1].stop_reason == "error"


async def test_agent_preserves_invalid_tool_call_for_recovery() -> None:
    provider = FakeProvider(
        [
            ToolCallDelta("call-1", "echo", '{"value":'),
            StreamEnd(stop_reason="tool_calls"),
            TextDelta("recovered"),
            StreamEnd(stop_reason="stop"),
        ]
    )
    agent = Agent(provider, ToolRegistry(), max_turns=2)

    events = [event async for event in agent.run("recover invalid arguments")]

    assert [event.text for event in events if isinstance(event, TextDelta)] == ["recovered"]
    assert provider.requests[1][-2].tool_calls[0].id == "call-1"
    assert "invalid JSON arguments" in provider.requests[1][-1].content

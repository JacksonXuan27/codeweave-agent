from codeweave import Agent
from codeweave.models import StreamEnd, TextDelta, ToolCall, ToolCallComplete
from codeweave.permissions import PermissionChecker, PermissionMode
from codeweave.providers import FakeProvider
from codeweave.tools import ToolRegistry, ToolSpec


def test_permission_modes_protect_mutating_tools() -> None:
    mutating = ToolSpec(
        "write",
        "Write",
        lambda arguments: "ok",
        {"type": "object"},
        mutating=True,
    )
    read_only = ToolSpec("read", "Read", lambda arguments: "ok", {"type": "object"})

    denied = PermissionChecker(PermissionMode.READ_ONLY).check(
        mutating,
        ToolCall("1", "write", {}),
    )

    assert denied.allowed is False
    assert "read-only" in denied.reason
    assert PermissionChecker(PermissionMode.READ_ONLY).allows(read_only, ToolCall("2", "read", {}))
    assert PermissionChecker(PermissionMode.AUTO).allows(mutating, ToolCall("3", "write", {}))


def test_confirm_mode_requires_an_explicit_approval_callback() -> None:
    mutating = ToolSpec(
        "write",
        "Write",
        lambda arguments: "ok",
        {"type": "object"},
        mutating=True,
    )
    call = ToolCall("1", "write", {"path": "notes.txt"})
    approvals: list[tuple[str, str]] = []

    denied = PermissionChecker(PermissionMode.CONFIRM).check(mutating, call)
    approved = PermissionChecker(
        PermissionMode.CONFIRM,
        confirmer=lambda spec, requested_call: approvals.append(
            (spec.name, requested_call.id)
        )
        or True,
    ).check(mutating, call)

    assert denied.allowed is False
    assert "confirmation" in denied.reason
    assert approved.allowed is True
    assert approvals == [("write", "1")]


async def test_agent_returns_read_only_denial_to_the_provider() -> None:
    provider = FakeProvider(
        [
            ToolCallComplete(ToolCall("call-1", "write", {"path": "notes.txt"})),
            StreamEnd(stop_reason="tool_calls"),
            TextDelta("handled"),
            StreamEnd(stop_reason="stop"),
        ]
    )
    tools = ToolRegistry()
    executions: list[dict[str, object]] = []
    tools.register(
        ToolSpec(
            "write",
            "Write a file",
            lambda arguments: executions.append(arguments) or "written",
            {"type": "object"},
            mutating=True,
        )
    )
    agent = Agent(
        provider,
        tools,
        permission_checker=PermissionChecker(PermissionMode.READ_ONLY),
        max_turns=2,
    )

    events = [event async for event in agent.run("write notes")]

    assert [event.text for event in events if isinstance(event, TextDelta)] == ["handled"]
    assert executions == []
    assert "read-only" in provider.requests[1][-1].content


async def test_agent_executes_mutating_tools_in_auto_mode() -> None:
    provider = FakeProvider(
        [
            ToolCallComplete(ToolCall("call-1", "write", {"path": "notes.txt"})),
            StreamEnd(stop_reason="tool_calls"),
            TextDelta("done"),
            StreamEnd(stop_reason="stop"),
        ]
    )
    tools = ToolRegistry()
    executions: list[dict[str, object]] = []
    tools.register(
        ToolSpec(
            "write",
            "Write a file",
            lambda arguments: executions.append(arguments) or "written",
            {"type": "object"},
            mutating=True,
        )
    )
    agent = Agent(
        provider,
        tools,
        permission_checker=PermissionChecker(PermissionMode.AUTO),
        max_turns=2,
    )

    events = [event async for event in agent.run("write notes")]

    assert [event.text for event in events if isinstance(event, TextDelta)] == ["done"]
    assert executions == [{"path": "notes.txt"}]
    assert provider.requests[1][-1].content == "written"

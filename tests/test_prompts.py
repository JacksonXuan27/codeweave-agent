from codeweave import Agent
from codeweave.models import Role, StreamEnd
from codeweave.prompts import PromptComposer
from codeweave.providers import FakeProvider
from codeweave.tools import ToolRegistry, ToolSpec


def test_prompt_composer_includes_tools_workspace_task_and_context() -> None:
    prompt = PromptComposer("/tmp/project").compose(
        tool_descriptions=["read_file: read a file"],
        task="Fix the bug",
        runtime_context={"branch": "main", "language": "python"},
    )

    assert "/tmp/project" in prompt
    assert "read_file" in prompt
    assert "Fix the bug" in prompt
    assert "branch: main" in prompt
    assert "language: python" in prompt


def test_prompt_composer_marks_an_empty_tool_registry() -> None:
    prompt = PromptComposer("/tmp/project").compose(
        tool_descriptions=[],
        task="Explain the codebase",
    )

    assert "No tools are currently available" in prompt


async def test_agent_injects_composed_system_prompt_before_user_message() -> None:
    provider = FakeProvider([StreamEnd(stop_reason="stop")])
    tools = ToolRegistry()
    tools.register(
        ToolSpec(
            "read_file",
            "Read a UTF-8 file.",
            lambda arguments: "",
            {"type": "object"},
        )
    )
    agent = Agent(
        provider,
        tools,
        prompt_composer=PromptComposer("/workspace"),
    )

    events = [
        event
        async for event in agent.run("Inspect README", runtime_context={"branch": "main"})
    ]

    assert events[-1].stop_reason == "stop"
    request = provider.requests[0]
    assert [message.role for message in request] == [Role.SYSTEM, Role.USER]
    assert "read_file" in request[0].content
    assert "/workspace" in request[0].content
    assert "branch: main" in request[0].content
    assert request[1].content == "Inspect README"

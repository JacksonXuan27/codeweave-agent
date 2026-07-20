from codeweave.models import Message, StreamEnd, TextDelta, ToolCall
from codeweave.providers import FakeProvider, create_provider


def test_message_and_tool_call_are_serializable() -> None:
    message = Message.user("inspect this file")
    call = ToolCall(id="call-1", name="read_file", arguments={"path": "README.md"})

    assert message.role == "user"
    assert call.arguments["path"] == "README.md"
    assert message.to_dict() == {"role": "user", "content": "inspect this file"}


async def test_fake_provider_streams_normalized_events() -> None:
    provider = FakeProvider(["hello", StreamEnd(stop_reason="stop")])

    events = [event async for event in provider.stream([Message.user("hi")])]

    assert isinstance(events[0], TextDelta)
    assert isinstance(events[-1], StreamEnd)


def test_provider_factory_supports_all_modes() -> None:
    assert create_provider("openai").mode == "openai"
    assert create_provider("anthropic").mode == "anthropic"
    assert create_provider("openai-compatible").mode == "openai-compatible"

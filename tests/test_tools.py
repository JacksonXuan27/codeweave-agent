from pathlib import Path

import pytest

from codeweave.models import ToolCall
from codeweave.tools import ToolRegistry, build_builtin_registry


def test_builtin_registry_reads_and_writes_within_workspace(tmp_path: Path) -> None:
    registry = build_builtin_registry(tmp_path)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")

    read_result = registry.execute(ToolCall("1", "read_file", {"path": "notes.txt"}))
    write_result = registry.execute(
        ToolCall("2", "write_file", {"path": "out.txt", "content": "world"})
    )

    assert read_result.ok is True
    assert read_result.output == "hello"
    assert write_result.ok is True
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "world"


def test_file_tool_rejects_workspace_escape(tmp_path: Path) -> None:
    registry = build_builtin_registry(tmp_path)

    result = registry.execute(ToolCall("1", "read_file", {"path": "../secret.txt"}))

    assert result.ok is False
    assert "workspace" in result.error.lower()


def test_unknown_tool_is_reported() -> None:
    registry = ToolRegistry()

    result = registry.execute(ToolCall("1", "missing", {}))

    assert result.ok is False
    assert "unknown tool" in result.error.lower()

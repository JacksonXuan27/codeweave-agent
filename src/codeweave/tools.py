from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from codeweave.models import ToolCall, ToolResult


ToolHandler = Callable[[dict[str, Any]], str]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    handler: ToolHandler
    parameters: dict[str, Any]
    mutating: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.schema() for spec in self.specs()]

    def execute(self, call: ToolCall) -> ToolResult:
        spec = self.get(call.name)
        if spec is None:
            return ToolResult(call.id, False, error=f"unknown tool: {call.name}")
        try:
            return ToolResult(call.id, True, output=spec.handler(call.arguments))
        except Exception as exc:
            return ToolResult(call.id, False, error=str(exc))


def build_builtin_registry(workspace: Path) -> ToolRegistry:
    root = workspace.expanduser().resolve()
    registry = ToolRegistry()

    def safe_path(arguments: dict[str, Any]) -> Path:
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("path must be a non-empty string")
        candidate = (root / raw_path).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("path escapes workspace")
        return candidate

    def read_file(arguments: dict[str, Any]) -> str:
        path = safe_path(arguments)
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {arguments['path']}")
        return path.read_text(encoding="utf-8")

    def write_file(arguments: dict[str, Any]) -> str:
        path = safe_path(arguments)
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {path.relative_to(root)}"

    def list_files(arguments: dict[str, Any]) -> str:
        path = safe_path({"path": arguments.get("path", ".")})
        if not path.is_dir():
            raise NotADirectoryError(f"directory not found: {arguments.get('path', '.')}")
        entries = sorted(item.relative_to(root).as_posix() for item in path.iterdir())
        return "\n".join(entries)

    def diff_file(arguments: dict[str, Any]) -> str:
        path = safe_path(arguments)
        old = arguments.get("old", "")
        new = arguments.get("new", "")
        if not isinstance(old, str) or not isinstance(new, str):
            raise ValueError("old and new must be strings")
        return "".join(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=str(path),
                tofile=str(path),
            )
        )

    text_parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    registry.register(ToolSpec("read_file", "Read a UTF-8 file.", read_file, text_parameters))
    registry.register(
        ToolSpec(
            "write_file",
            "Write UTF-8 content to a file.",
            write_file,
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            mutating=True,
        )
    )
    registry.register(
        ToolSpec(
            "list_files",
            "List files in a workspace directory.",
            list_files,
            {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
            },
        )
    )
    registry.register(
        ToolSpec(
            "diff_file",
            "Show a unified diff between two strings.",
            diff_file,
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        )
    )
    return registry

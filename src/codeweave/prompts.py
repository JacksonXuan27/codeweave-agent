from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path


DEFAULT_SYSTEM_INSTRUCTIONS = """You are CodeWeave Agent, a terminal coding assistant.
Work carefully within the configured workspace, inspect relevant files before editing,
and use available tools only when they help complete the task."""


class PromptComposer:
    def __init__(
        self,
        workspace: str | Path,
        *,
        system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
    ) -> None:
        instructions = system_instructions.strip()
        if not instructions:
            raise ValueError("system_instructions must be non-empty")
        self.workspace = Path(workspace).expanduser()
        self.system_instructions = instructions

    def compose(
        self,
        *,
        tool_descriptions: Iterable[str],
        task: str,
        runtime_context: Mapping[str, object] | None = None,
    ) -> str:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        tools = [description.strip() for description in tool_descriptions if description.strip()]
        tool_section = "\n".join(f"- {description}" for description in tools)
        if not tool_section:
            tool_section = "- No tools are currently available."

        context_section = "- No additional runtime context."
        if runtime_context:
            context_items = sorted(runtime_context.items(), key=lambda item: str(item[0]))
            context_section = "\n".join(
                f"- {key}: {value}" for key, value in context_items
            )

        return "\n\n".join(
            [
                f"## System Instructions\n{self.system_instructions}",
                f"## Environment\n- Workspace: {self.workspace.as_posix()}",
                f"## Available Tools\n{tool_section}",
                f"## Runtime Context\n{context_section}",
                f"## User Task\n{task.strip()}",
            ]
        )

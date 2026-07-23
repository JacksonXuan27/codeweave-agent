from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from codeweave.models import ToolCall
from codeweave.tools import ToolSpec


class PermissionMode(StrEnum):
    READ_ONLY = "read-only"
    CONFIRM = "confirm"
    AUTO = "auto"


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    allowed: bool
    reason: str = ""


ConfirmationHandler = Callable[[ToolSpec, ToolCall], bool]


class PermissionChecker:
    def __init__(
        self,
        mode: PermissionMode | str = PermissionMode.CONFIRM,
        *,
        confirmer: ConfirmationHandler | None = None,
    ) -> None:
        self.mode = PermissionMode(mode)
        self.confirmer = confirmer

    def check(self, spec: ToolSpec, call: ToolCall) -> PermissionDecision:
        if not spec.mutating:
            return PermissionDecision(True)
        if self.mode is PermissionMode.AUTO:
            return PermissionDecision(True)
        if self.mode is PermissionMode.READ_ONLY:
            return PermissionDecision(False, f"{spec.name} is blocked in read-only mode")
        if self.confirmer is None:
            return PermissionDecision(False, f"{spec.name} requires confirmation")

        try:
            approved = self.confirmer(spec, call)
        except Exception as exc:
            return PermissionDecision(False, f"confirmation failed: {exc}")
        if approved:
            return PermissionDecision(True)
        return PermissionDecision(False, f"confirmation denied {spec.name}")

    def allows(self, spec: ToolSpec, call: ToolCall) -> bool:
        return self.check(spec, call).allowed

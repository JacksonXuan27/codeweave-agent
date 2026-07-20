from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when configuration values cannot be parsed."""


@dataclass(slots=True)
class AppConfig:
    provider_mode: str = "openai-compatible"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    permission_mode: str = "confirm"
    max_turns: int = 20
    max_tokens: int = 4096
    workspace: Path = Path.cwd()


def load_config(environ: dict[str, str] | None = None) -> AppConfig:
    values = os.environ if environ is None else environ
    try:
        max_turns = int(values.get("CODEWEAVE_MAX_TURNS", "20"))
        max_tokens = int(values.get("CODEWEAVE_MAX_TOKENS", "4096"))
    except ValueError as exc:
        raise ConfigError("CODEWEAVE_MAX_TURNS and CODEWEAVE_MAX_TOKENS must be integers") from exc
    if max_turns < 1 or max_tokens < 1:
        raise ConfigError("turn and token limits must be positive")

    provider_mode = values.get("CODEWEAVE_PROVIDER_MODE", "openai-compatible")
    allowed_modes = {"openai", "anthropic", "openai-compatible"}
    if provider_mode not in allowed_modes:
        raise ConfigError(f"unsupported provider mode: {provider_mode}")

    permission_mode = values.get("CODEWEAVE_PERMISSION_MODE", "confirm")
    if permission_mode not in {"read-only", "confirm", "auto"}:
        raise ConfigError(f"unsupported permission mode: {permission_mode}")

    workspace = Path(values.get("CODEWEAVE_WORKSPACE", Path.cwd())).expanduser().resolve()
    return AppConfig(
        provider_mode=provider_mode,
        model=values.get("CODEWEAVE_MODEL", "gpt-4o-mini"),
        api_key=values.get("CODEWEAVE_API_KEY"),
        base_url=values.get("CODEWEAVE_BASE_URL"),
        permission_mode=permission_mode,
        max_turns=max_turns,
        max_tokens=max_tokens,
        workspace=workspace,
    )

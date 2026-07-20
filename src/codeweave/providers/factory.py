from __future__ import annotations

from codeweave.models import ProviderConfig
from codeweave.providers.anthropic import AnthropicProvider
from codeweave.providers.base import LLMProvider, ProviderError
from codeweave.providers.fake import FakeProvider
from codeweave.providers.openai import OpenAIProvider


def create_provider(
    mode: str,
    *,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMProvider:
    if mode == "fake":
        return FakeProvider()
    config = ProviderConfig(mode=mode, model=model, api_key=api_key, base_url=base_url)
    if mode in {"openai", "openai-compatible"}:
        return OpenAIProvider(config)
    if mode == "anthropic":
        return AnthropicProvider(config)
    raise ProviderError(f"unsupported provider mode: {mode}")

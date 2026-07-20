from codeweave.config import AppConfig, load_config


def test_default_config_has_safe_defaults() -> None:
    config = AppConfig()

    assert config.provider_mode == "openai-compatible"
    assert config.permission_mode == "confirm"
    assert config.max_turns == 20


def test_load_config_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("CODEWEAVE_PROVIDER_MODE", "anthropic")
    monkeypatch.setenv("CODEWEAVE_MODEL", "claude-test")
    monkeypatch.setenv("CODEWEAVE_MAX_TURNS", "7")

    config = load_config()

    assert config.provider_mode == "anthropic"
    assert config.model == "claude-test"
    assert config.max_turns == 7

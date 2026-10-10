import pytest

from miresolver.models.settings import (
    ModelConfigError,
    TierSettings,
    load_model_settings,
)

YAML = """
tiers:
  embed:
    provider: openai-compatible
    base_url: http://internal-llm:8000/v1
    model: some-embedding-model
    api_key_env: INTERNAL_LLM_KEY
  screen:
    provider: anthropic
    model: some-small-model
  investigate:
    provider: anthropic
    model: some-mid-model
  synthesize:
    provider: gemini
    model: some-large-model
"""


def test_loads_every_tier():
    settings = load_model_settings(YAML)
    assert settings.tier("embed") == TierSettings(
        provider="openai-compatible",
        model="some-embedding-model",
        base_url="http://internal-llm:8000/v1",
        api_key_env="INTERNAL_LLM_KEY",
    )


def test_optional_fields_default_to_none():
    settings = load_model_settings(YAML)
    screen = settings.tier("screen")
    assert screen.base_url is None
    assert screen.api_key_env is None


def test_unknown_tier_raises():
    settings = load_model_settings(YAML)
    with pytest.raises(ModelConfigError, match="unknown tier"):
        settings.tier("nonexistent")


def test_missing_tier_in_file_raises():
    with pytest.raises(ModelConfigError, match="investigate"):
        load_model_settings(
            "tiers:\n"
            "  embed: {provider: anthropic, model: m}\n"
            "  screen: {provider: anthropic, model: m}\n"
            "  synthesize: {provider: anthropic, model: m}\n"
        )


def test_missing_provider_raises():
    with pytest.raises(ModelConfigError, match="provider"):
        load_model_settings("tiers:\n  embed: {model: m}\n")


def test_missing_model_raises():
    with pytest.raises(ModelConfigError, match="model"):
        load_model_settings("tiers:\n  embed: {provider: anthropic}\n")


def test_empty_document_raises():
    with pytest.raises(ModelConfigError, match="tiers"):
        load_model_settings("")


def test_list_document_raises():
    with pytest.raises(ModelConfigError, match="top level"):
        load_model_settings("- item1\n- item2\n")


def test_string_document_raises():
    with pytest.raises(ModelConfigError, match="top level"):
        load_model_settings("just a string")


def test_tiers_mapping_is_immutable():
    settings = load_model_settings(YAML)
    with pytest.raises(TypeError):
        settings.tiers["embed"] = TierSettings(
            provider="anthropic", model="new-model"
        )

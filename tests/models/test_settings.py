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


PRICED_YAML = """
tiers:
  embed: {provider: openai-compatible, base_url: "http://x/v1", model: e, api_key_env: K}
  screen:
    provider: openai-compatible
    base_url: "http://x/v1"
    model: s
    api_key_env: K
    input_cost_per_mtok: 0.15
    output_cost_per_mtok: 0.6
  investigate: {provider: anthropic, model: i, api_key_env: K}
  synthesize: {provider: gemini, model: y, api_key_env: K}
"""


def test_prices_are_optional_and_parsed_as_floats():
    settings = load_model_settings(PRICED_YAML)
    assert settings.tier("screen").input_cost_per_mtok == 0.15
    assert settings.tier("screen").output_cost_per_mtok == 0.6
    assert settings.tier("investigate").input_cost_per_mtok is None


def test_cost_is_none_without_prices_and_dollars_with_them():
    from miresolver.models.chat import Usage

    settings = load_model_settings(PRICED_YAML)
    usage = Usage(input_tokens=2_000_000, output_tokens=1_000_000)
    assert settings.tier("investigate").cost(usage) is None
    assert settings.tier("screen").cost(usage) == pytest.approx(0.9)


def test_non_numeric_price_is_a_config_error():
    bad = PRICED_YAML.replace("0.15", "cheap")
    with pytest.raises(ModelConfigError, match="input_cost_per_mtok"):
        load_model_settings(bad)

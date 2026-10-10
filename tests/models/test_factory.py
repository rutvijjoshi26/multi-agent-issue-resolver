import asyncio

import pytest

from miresolver.models.anthropic import AnthropicEmbedder
from miresolver.models.embedding import EmbeddingError, OpenAICompatibleEmbedder
from miresolver.models.factory import PROVIDERS, build_embedder
from miresolver.models.gemini import GeminiEmbedder
from miresolver.models.settings import ModelConfigError, load_model_settings

def settings_for(provider: str, *, base_url: str | None = None) -> str:
    url = f"\n    base_url: {base_url}" if base_url else ""
    return (
        "tiers:\n"
        f"  embed:\n    provider: {provider}\n    model: m{url}\n"
        "    api_key_env: TEST_KEY\n"
        "  screen: {provider: anthropic, model: m, api_key_env: TEST_KEY}\n"
        "  investigate: {provider: anthropic, model: m, api_key_env: TEST_KEY}\n"
        "  synthesize: {provider: anthropic, model: m, api_key_env: TEST_KEY}\n"
    )


ENV = {"TEST_KEY": "secret"}


def test_builds_an_openai_compatible_embedder():
    settings = load_model_settings(
        settings_for("openai-compatible", base_url="http://host/v1")
    )
    embedder = build_embedder(settings, ENV)
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    asyncio.run(embedder.aclose())


def test_anthropic_embedder_refuses_clearly_rather_than_guessing():
    settings = load_model_settings(settings_for("anthropic"))
    embedder = build_embedder(settings, ENV)
    with pytest.raises(EmbeddingError, match="anthropic"):
        asyncio.run(embedder.embed(["hello"]))


def test_gemini_embedder_refuses_clearly_rather_than_guessing():
    settings = load_model_settings(settings_for("gemini"))
    embedder = build_embedder(settings, ENV)
    with pytest.raises(EmbeddingError, match="gemini"):
        asyncio.run(embedder.embed(["hello"]))


def test_unknown_provider_raises():
    settings = load_model_settings(settings_for("not-a-provider"))
    with pytest.raises(ModelConfigError, match="not-a-provider"):
        build_embedder(settings, ENV)


def test_missing_api_key_in_environment_raises():
    settings = load_model_settings(
        settings_for("openai-compatible", base_url="http://host/v1")
    )
    with pytest.raises(ModelConfigError, match="TEST_KEY"):
        build_embedder(settings, {})


def test_provider_set_is_what_the_factory_dispatches_on():
    assert PROVIDERS == frozenset({"openai-compatible", "anthropic", "gemini"})


_EXPECTED_ADAPTER = {
    "openai-compatible": OpenAICompatibleEmbedder,
    "anthropic": AnthropicEmbedder,
    "gemini": GeminiEmbedder,
}


@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_every_provider_dispatches_to_its_own_adapter_not_a_fallback(provider):
    base_url = "http://host/v1" if provider == "openai-compatible" else None
    settings = load_model_settings(settings_for(provider, base_url=base_url))
    embedder = build_embedder(settings, ENV)
    assert type(embedder) is _EXPECTED_ADAPTER[provider]


def chat_settings(provider: str, *, base_url: str | None = None):
    url = f", base_url: '{base_url}'" if base_url else ""
    return load_model_settings(
        "tiers:\n"
        "  embed: {provider: openai-compatible, base_url: 'http://h/v1', model: e, api_key_env: TEST_KEY}\n"
        f"  screen: {{provider: {provider}, model: s, api_key_env: TEST_KEY{url}}}\n"
        "  investigate: {provider: anthropic, model: i, api_key_env: TEST_KEY}\n"
        "  synthesize: {provider: gemini, model: y, api_key_env: TEST_KEY}\n"
    )


def test_build_chat_picks_the_adapter_named_by_the_tier():
    from miresolver.models.anthropic import AnthropicChat
    from miresolver.models.chat import OpenAICompatibleChat
    from miresolver.models.factory import build_chat
    from miresolver.models.gemini import GeminiChat

    settings = chat_settings("openai-compatible", base_url="http://h/v1")
    clients = [build_chat(settings, t, ENV) for t in ("screen", "investigate", "synthesize")]
    assert [type(c) for c in clients] == [OpenAICompatibleChat, AnthropicChat, GeminiChat]
    for c in clients:
        asyncio.run(c.aclose())


def test_build_chat_rejects_the_embed_tier_and_unknown_providers():
    from miresolver.models.factory import build_chat

    settings = chat_settings("mystery")
    with pytest.raises(ModelConfigError, match="embed"):
        build_chat(settings, "embed", ENV)
    with pytest.raises(ModelConfigError, match="mystery"):
        build_chat(settings, "screen", ENV)


def test_build_chat_requires_the_key():
    from miresolver.models.factory import build_chat

    settings = chat_settings("anthropic")
    with pytest.raises(ModelConfigError, match="TEST_KEY"):
        build_chat(settings, "screen", {})

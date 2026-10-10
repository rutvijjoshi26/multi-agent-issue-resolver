from collections.abc import Mapping

import httpx

from miresolver.models.anthropic import AnthropicChat, AnthropicEmbedder
from miresolver.models.chat import ChatClient, OpenAICompatibleChat
from miresolver.models.embedding import EmbeddingClient, OpenAICompatibleEmbedder
from miresolver.models.gemini import GeminiChat, GeminiEmbedder
from miresolver.models.settings import ModelConfigError, ModelSettings, TierSettings

PROVIDERS: frozenset[str] = frozenset(
    {"openai-compatible", "anthropic", "gemini"}
)


def _api_key(tier: TierSettings, env: Mapping[str, str]) -> str:
    if not tier.api_key_env:
        raise ModelConfigError(
            f"tier using provider {tier.provider} is missing api_key_env"
        )
    key = env.get(tier.api_key_env)
    if not key:
        raise ModelConfigError(
            f"environment variable {tier.api_key_env} is not set"
        )
    return key


def build_embedder(
    settings: ModelSettings,
    env: Mapping[str, str],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> EmbeddingClient:
    tier = settings.tier("embed")
    key = _api_key(tier, env)
    match tier.provider:
        case "openai-compatible":
            return OpenAICompatibleEmbedder(tier, key, transport=transport)
        case "anthropic":
            return AnthropicEmbedder(tier, key)
        case "gemini":
            return GeminiEmbedder(tier, key)
        case _:
            raise ModelConfigError(f"unknown provider: {tier.provider}")


CHAT_TIERS: tuple[str, ...] = ("screen", "investigate", "synthesize")


def build_chat(
    settings: ModelSettings,
    tier_name: str,
    env: Mapping[str, str],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ChatClient:
    """Build the chat client a tier's config names.

    Only the agent tiers are chat tiers; the embed tier serves vectors.
    """
    if tier_name not in CHAT_TIERS:
        raise ModelConfigError(
            f"tier {tier_name} is not a chat tier; expected one of "
            f"{', '.join(CHAT_TIERS)}"
        )
    tier = settings.tier(tier_name)
    if tier.provider not in PROVIDERS:
        raise ModelConfigError(f"unknown provider: {tier.provider}")
    key = _api_key(tier, env)
    match tier.provider:
        case "openai-compatible":
            return OpenAICompatibleChat(tier, key, transport=transport)
        case "anthropic":
            return AnthropicChat(tier, key, transport=transport)
        case _:
            return GeminiChat(tier, key, transport=transport)

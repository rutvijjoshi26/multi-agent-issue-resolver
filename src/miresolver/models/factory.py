from collections.abc import Mapping

import httpx

from miresolver.models.anthropic import AnthropicEmbedder
from miresolver.models.embedding import EmbeddingClient, OpenAICompatibleEmbedder
from miresolver.models.gemini import GeminiEmbedder
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

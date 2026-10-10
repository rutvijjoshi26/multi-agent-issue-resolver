from collections.abc import Sequence

from miresolver.models.embedding import EmbeddingError
from miresolver.models.settings import TierSettings


class AnthropicEmbedder:
    """Placeholder so the embed tier fails loudly rather than silently.

    Anthropic does not serve a first-party embeddings endpoint. Point the
    embed tier at an openai-compatible endpoint instead; every other tier
    can still be anthropic.
    """

    def __init__(self, settings: TierSettings, api_key: str) -> None:
        self._settings = settings

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbeddingError(
            "provider anthropic cannot serve the embed tier; point embed at an "
            "openai-compatible endpoint and keep anthropic for the other tiers"
        )

    async def aclose(self) -> None:
        return None

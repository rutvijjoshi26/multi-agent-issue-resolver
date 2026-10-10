from collections.abc import Sequence

from miresolver.models.embedding import EmbeddingError
from miresolver.models.settings import TierSettings


class GeminiEmbedder:
    """Placeholder so the embed tier fails loudly rather than silently.

    Gemini's embedding API is not OpenAI-shaped. Point the embed tier at an
    openai-compatible endpoint, or at Gemini's own OpenAI compatibility
    endpoint, and keep gemini for the other tiers.
    """

    def __init__(self, settings: TierSettings, api_key: str) -> None:
        self._settings = settings

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbeddingError(
            "provider gemini cannot serve the embed tier directly; point embed "
            "at an openai-compatible endpoint and keep gemini for other tiers"
        )

    async def aclose(self) -> None:
        return None

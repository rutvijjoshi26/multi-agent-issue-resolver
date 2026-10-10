import json
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from miresolver.models.settings import TierSettings


class EmbeddingError(Exception):
    """Raised when an embedding provider cannot produce vectors."""


class EmbeddingClient(Protocol):
    """What the retrieval layer needs from any embedding provider.

    Implemented by the adapters in this package and by test doubles.
    """

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


def parse_embedding_response(payload: dict[str, Any]) -> list[list[float]]:
    data = payload.get("data")
    if not data:
        raise EmbeddingError(f"embedding response contained no data: {payload}")
    for item in data:
        if "index" not in item:
            raise EmbeddingError(f"embedding item missing 'index': {item}")
        if "embedding" not in item:
            raise EmbeddingError(f"embedding item missing 'embedding': {item}")
    ordered = sorted(data, key=lambda item: item["index"])
    return [list(item["embedding"]) for item in ordered]


class OpenAICompatibleEmbedder:
    """Talks to any endpoint implementing the OpenAI embeddings API."""

    def __init__(
        self,
        settings: TierSettings,
        api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not settings.base_url:
            raise EmbeddingError(
                f"provider {settings.provider} requires base_url in config"
            )
        self._model = settings.model
        self._client = httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
            timeout=60.0,
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.post(
                "/embeddings", json={"model": self._model, "input": list(texts)}
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embedding request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise EmbeddingError(
                f"embedding response was not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise EmbeddingError(
                f"embedding response was not a JSON object: {payload!r}"
            )
        return parse_embedding_response(payload)

    async def aclose(self) -> None:
        await self._client.aclose()

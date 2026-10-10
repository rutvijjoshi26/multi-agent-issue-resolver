from collections.abc import Mapping, Sequence

import httpx

from miresolver.models.chat import (
    DEFAULT_TIMEOUT_S,
    ChatError,
    ChatResult,
    Usage,
    post_json,
    shape_errors_as_chat_errors,
)
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


DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
JSON_INSTRUCTION = (
    "\n\nRespond with a single JSON object and nothing else: no prose, no "
    "code fences."
)


class AnthropicChat:
    """Native Anthropic Messages API adapter.

    The Messages API has no generic JSON switch, so json_mode appends an
    instruction to the system prompt and callers parse tolerantly with
    extract_json.
    """

    def __init__(
        self,
        settings: TierSettings,
        api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = settings.model
        self._client = httpx.AsyncClient(
            base_url=(settings.base_url or DEFAULT_ANTHROPIC_URL).rstrip("/"),
            headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
            transport=transport,
            timeout=DEFAULT_TIMEOUT_S,
        )

    async def complete(
        self,
        system: str,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> ChatResult:
        body = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system + (JSON_INSTRUCTION if json_mode else ""),
            "messages": [dict(m) for m in messages],
        }
        payload = await post_json(self._client, "/v1/messages", body)
        blocks = payload.get("content")
        if not isinstance(blocks, list) or not blocks:
            raise ChatError(f"anthropic response contained no content: {payload}")
        with shape_errors_as_chat_errors():
            text = "".join(
                b.get("text", "") for b in blocks if b.get("type") == "text"
            )
            usage = payload.get("usage") or {}
            return ChatResult(
                text=text,
                usage=Usage(
                    int(usage.get("input_tokens") or 0),
                    int(usage.get("output_tokens") or 0),
                ),
            )

    async def aclose(self) -> None:
        await self._client.aclose()

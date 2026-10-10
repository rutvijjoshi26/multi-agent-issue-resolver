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


DEFAULT_GEMINI_URL = "https://generativelanguage.googleapis.com"


class GeminiChat:
    """Native Gemini generateContent adapter."""

    def __init__(
        self,
        settings: TierSettings,
        api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = settings.model
        self._client = httpx.AsyncClient(
            base_url=(settings.base_url or DEFAULT_GEMINI_URL).rstrip("/"),
            headers={"x-goog-api-key": api_key},
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
        config: dict[str, object] = {"maxOutputTokens": max_tokens}
        if json_mode:
            config["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {
                    "role": "model" if m["role"] == "assistant" else "user",
                    "parts": [{"text": m["content"]}],
                }
                for m in messages
            ],
            "generationConfig": config,
        }
        payload = await post_json(
            self._client, f"/v1beta/models/{self._model}:generateContent", body
        )
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ChatError(f"gemini response contained no candidates: {payload}")
        with shape_errors_as_chat_errors():
            parts = (candidates[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts)
            usage = payload.get("usageMetadata") or {}
            # Thinking models bill their reasoning tokens as output.
            output = int(usage.get("candidatesTokenCount") or 0) + int(
                usage.get("thoughtsTokenCount") or 0
            )
            return ChatResult(
                text=text,
                usage=Usage(int(usage.get("promptTokenCount") or 0), output),
            )

    async def aclose(self) -> None:
        await self._client.aclose()

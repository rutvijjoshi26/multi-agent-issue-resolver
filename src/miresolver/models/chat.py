import contextlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from miresolver.models.settings import TierSettings

DEFAULT_TIMEOUT_S = 120.0


class ChatError(Exception):
    """Raised when a chat provider cannot produce a usable reply."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class ChatResult:
    text: str
    usage: Usage


class ChatClient(Protocol):
    """What the agents need from any chat provider.

    The system prompt is separate from the message list because two of the
    three supported wire formats carry it outside the messages.
    """

    async def complete(
        self,
        system: str,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> ChatResult: ...

    async def aclose(self) -> None: ...


async def post_json(
    client: httpx.AsyncClient, path: str, body: Mapping[str, Any]
) -> dict[str, Any]:
    """POST a JSON body and return a JSON object, or raise ChatError."""
    try:
        response = await client.post(path, json=dict(body))
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        raise ChatError(f"chat request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ChatError(f"chat response was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ChatError(f"chat response was not a JSON object: {payload!r}")
    return payload


@contextlib.contextmanager
def shape_errors_as_chat_errors() -> Iterator[None]:
    """Turn a reply of an unexpected shape into ChatError, so callers that
    degrade gracefully on ChatError also do so for malformed payloads."""
    try:
        yield
    except (AttributeError, TypeError, KeyError, IndexError, ValueError) as exc:
        raise ChatError(f"chat response had an unexpected shape: {exc}") from exc


def _int(value: object) -> int:
    return value if isinstance(value, int) else 0


class OpenAICompatibleChat:
    """Talks to any endpoint implementing the OpenAI chat completions API."""

    def __init__(
        self,
        settings: TierSettings,
        api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not settings.base_url:
            raise ChatError(
                f"provider {settings.provider} requires base_url in config"
            )
        self._model = settings.model
        self._client = httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
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
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}]
            + [dict(m) for m in messages],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        payload = await post_json(self._client, "/chat/completions", body)
        choices = payload.get("choices")
        if not choices:
            raise ChatError(f"chat response contained no choices: {payload}")
        with shape_errors_as_chat_errors():
            text = (choices[0].get("message") or {}).get("content") or ""
            usage = payload.get("usage") or {}
            return ChatResult(
                text=str(text),
                usage=Usage(
                    _int(usage.get("prompt_tokens")),
                    _int(usage.get("completion_tokens")),
                ),
            )

    async def aclose(self) -> None:
        await self._client.aclose()


_DECODER = json.JSONDecoder()


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply.

    Models asked for JSON still wrap it in code fences or a sentence of
    prose often enough that a strict parse would turn usable answers into
    failures. Scanning for the first decodable object handles both.
    """
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ChatError(f"no JSON object in model reply: {text[:200]!r}")

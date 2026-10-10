import asyncio
import json

import httpx
import pytest

from miresolver.models.chat import (
    ChatError,
    OpenAICompatibleChat,
    Usage,
    extract_json,
)
from miresolver.models.settings import TierSettings

SETTINGS = TierSettings(
    provider="openai-compatible",
    model="test-chat-model",
    base_url="http://test-host/v1/",
    api_key_env="TEST_KEY",
)

OK = {
    "choices": [{"message": {"role": "assistant", "content": "hello"}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}


def _run(handler, **kwargs):
    seen: dict = {}

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return handler(request)

    async def go():
        chat = OpenAICompatibleChat(
            SETTINGS, "secret", transport=httpx.MockTransport(wrapped)
        )
        try:
            return await chat.complete(
                "be brief", [{"role": "user", "content": "hi"}], **kwargs
            )
        finally:
            await chat.aclose()

    return asyncio.run(go()), seen


def test_posts_system_then_messages_to_chat_completions():
    result, seen = _run(lambda r: httpx.Response(200, json=OK), max_tokens=50)
    assert seen["url"] == "http://test-host/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["model"] == "test-chat-model"
    assert seen["body"]["max_tokens"] == 50
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert "response_format" not in seen["body"]
    assert result.text == "hello"
    assert result.usage == Usage(input_tokens=12, output_tokens=3)


def test_json_mode_requests_a_json_object():
    _, seen = _run(lambda r: httpx.Response(200, json=OK), json_mode=True)
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_missing_usage_counts_as_zero():
    body = {"choices": [{"message": {"content": "x"}}]}
    result, _ = _run(lambda r: httpx.Response(200, json=body))
    assert result.usage == Usage()


def test_http_error_raises_chat_error():
    with pytest.raises(ChatError, match="chat request failed"):
        _run(lambda r: httpx.Response(500, json={"error": "boom"}))


def test_missing_choices_raises_chat_error():
    with pytest.raises(ChatError, match="choices"):
        _run(lambda r: httpx.Response(200, json={"usage": {}}))


def test_requires_base_url():
    with pytest.raises(ChatError, match="base_url"):
        OpenAICompatibleChat(
            TierSettings(provider="openai-compatible", model="m"), "k"
        )


def test_usage_adds():
    assert Usage(1, 2) + Usage(3, 4) == Usage(4, 6)


def test_extract_json_bare():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_extract_json_with_surrounding_prose():
    text = 'Sure, here it is: {"a": {"b": "}"}} hope that helps'
    assert extract_json(text) == {"a": {"b": "}"}}


def test_extract_json_raises_without_an_object():
    with pytest.raises(ChatError, match="no JSON object"):
        extract_json("I cannot help with that")


def test_extract_json_rejects_a_top_level_list():
    with pytest.raises(ChatError):
        extract_json("[1, 2]")

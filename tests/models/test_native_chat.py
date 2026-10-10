import asyncio
import json

import httpx
import pytest

from miresolver.models.anthropic import AnthropicChat
from miresolver.models.chat import ChatError, Usage
from miresolver.models.gemini import GeminiChat
from miresolver.models.settings import TierSettings

MESSAGES = [
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "reply"},
    {"role": "user", "content": "second"},
]


def _call(cls, settings, response, **kwargs):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return response

    async def go():
        chat = cls(settings, "secret", transport=httpx.MockTransport(handler))
        try:
            return await chat.complete("sys", MESSAGES, **kwargs)
        finally:
            await chat.aclose()

    return asyncio.run(go()), seen


ANTHROPIC = TierSettings(provider="anthropic", model="a-model", api_key_env="K")
ANTHROPIC_OK = {
    "content": [
        {"type": "text", "text": "part one "},
        {"type": "text", "text": "part two"},
    ],
    "usage": {"input_tokens": 20, "output_tokens": 5},
}


def test_anthropic_posts_native_messages_request():
    result, seen = _call(
        AnthropicChat, ANTHROPIC, httpx.Response(200, json=ANTHROPIC_OK),
        max_tokens=77,
    )
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "secret"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"] == {
        "model": "a-model",
        "max_tokens": 77,
        "system": "sys",
        "messages": MESSAGES,
    }
    assert result.text == "part one part two"
    assert result.usage == Usage(20, 5)


def test_anthropic_honours_a_base_url_override():
    settings = TierSettings(
        provider="anthropic", model="m", base_url="http://gateway/", api_key_env="K"
    )
    _, seen = _call(AnthropicChat, settings, httpx.Response(200, json=ANTHROPIC_OK))
    assert seen["url"] == "http://gateway/v1/messages"


def test_anthropic_json_mode_adds_an_instruction_to_the_system_prompt():
    _, seen = _call(
        AnthropicChat, ANTHROPIC, httpx.Response(200, json=ANTHROPIC_OK),
        json_mode=True,
    )
    assert seen["body"]["system"].startswith("sys")
    assert "JSON" in seen["body"]["system"]


def test_anthropic_errors_raise_chat_error():
    with pytest.raises(ChatError):
        _call(AnthropicChat, ANTHROPIC, httpx.Response(401, json={}))
    with pytest.raises(ChatError, match="content"):
        _call(AnthropicChat, ANTHROPIC, httpx.Response(200, json={"usage": {}}))


GEMINI = TierSettings(provider="gemini", model="g-model", api_key_env="K")
GEMINI_OK = {
    "candidates": [{"content": {"parts": [{"text": "hi "}, {"text": "there"}]}}],
    "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4},
}


def test_gemini_posts_native_generate_content_request():
    result, seen = _call(
        GeminiChat, GEMINI, httpx.Response(200, json=GEMINI_OK), max_tokens=33
    )
    assert seen["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "g-model:generateContent"
    )
    assert seen["headers"]["x-goog-api-key"] == "secret"
    body = seen["body"]
    assert body["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["contents"][1]["parts"] == [{"text": "reply"}]
    assert body["generationConfig"] == {"maxOutputTokens": 33}
    assert result.text == "hi there"
    assert result.usage == Usage(9, 4)


def test_gemini_json_mode_sets_the_response_mime_type():
    _, seen = _call(
        GeminiChat, GEMINI, httpx.Response(200, json=GEMINI_OK), json_mode=True
    )
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_errors_raise_chat_error():
    with pytest.raises(ChatError):
        _call(GeminiChat, GEMINI, httpx.Response(500, json={}))
    with pytest.raises(ChatError, match="candidates"):
        _call(GeminiChat, GEMINI, httpx.Response(200, json={"candidates": []}))


def test_gemini_counts_thinking_tokens_as_output():
    body = dict(GEMINI_OK, usageMetadata={"promptTokenCount": 9, "candidatesTokenCount": 4,
                                          "thoughtsTokenCount": 30})
    result, _ = _call(GeminiChat, GEMINI, httpx.Response(200, json=body))
    assert result.usage == Usage(9, 34)


def test_a_malformed_payload_shape_is_a_chat_error():
    with pytest.raises(ChatError):
        _call(AnthropicChat, ANTHROPIC, httpx.Response(200, json={"content": ["not a block"]}))
    with pytest.raises(ChatError):
        _call(GeminiChat, GEMINI, httpx.Response(200, json={"candidates": ["x"]}))

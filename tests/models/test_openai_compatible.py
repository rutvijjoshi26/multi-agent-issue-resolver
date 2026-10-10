import asyncio
import json

import httpx
import pytest

from miresolver.models.embedding import (
    EmbeddingError,
    OpenAICompatibleEmbedder,
    parse_embedding_response,
)
from miresolver.models.settings import TierSettings

SETTINGS = TierSettings(
    provider="openai-compatible",
    model="test-embedding-model",
    base_url="http://test-host/v1",
    api_key_env="TEST_KEY",
)


def test_parse_orders_vectors_by_index_not_response_order():
    payload = {
        "data": [
            {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            {"index": 0, "embedding": [0.1, 0.2, 0.3]},
        ]
    }
    assert parse_embedding_response(payload) == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_parse_raises_on_missing_data():
    with pytest.raises(EmbeddingError, match="no data"):
        parse_embedding_response({"error": {"message": "nope"}})


def test_parse_raises_on_item_missing_index():
    payload = {"data": [{"embedding": [0.1, 0.2]}]}
    with pytest.raises(EmbeddingError, match="index"):
        parse_embedding_response(payload)


def test_parse_raises_on_item_missing_embedding():
    payload = {"data": [{"index": 0}]}
    with pytest.raises(EmbeddingError, match="embedding"):
        parse_embedding_response(payload)


def test_embed_posts_to_the_configured_base_url_and_model():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]},
        )

    embedder = OpenAICompatibleEmbedder(
        SETTINGS, "sk-test", transport=httpx.MockTransport(handler)
    )
    vectors = asyncio.run(embedder.embed(["hello"]))
    asyncio.run(embedder.aclose())

    assert vectors == [[1.0, 2.0]]
    assert seen["url"] == "http://test-host/v1/embeddings"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"] == {"model": "test-embedding-model", "input": ["hello"]}


def test_embed_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    embedder = OpenAICompatibleEmbedder(
        SETTINGS, "sk-test", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(EmbeddingError):
        asyncio.run(embedder.embed(["hello"]))
    asyncio.run(embedder.aclose())


def test_embed_raises_on_non_json_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json", headers={"content-type": "text/plain"})

    embedder = OpenAICompatibleEmbedder(
        SETTINGS, "sk-test", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(EmbeddingError):
        asyncio.run(embedder.embed(["hello"]))
    asyncio.run(embedder.aclose())


def test_embed_raises_on_json_list_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    embedder = OpenAICompatibleEmbedder(
        SETTINGS, "sk-test", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(EmbeddingError):
        asyncio.run(embedder.embed(["hello"]))
    asyncio.run(embedder.aclose())


def test_embed_returns_empty_without_calling_the_api_for_empty_input():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the API for empty input")

    embedder = OpenAICompatibleEmbedder(
        SETTINGS, "sk-test", transport=httpx.MockTransport(handler)
    )
    assert asyncio.run(embedder.embed([])) == []
    asyncio.run(embedder.aclose())


def test_missing_base_url_raises_at_construction():
    settings = TierSettings(provider="openai-compatible", model="m")
    with pytest.raises(EmbeddingError, match="base_url"):
        OpenAICompatibleEmbedder(settings, "sk-test")

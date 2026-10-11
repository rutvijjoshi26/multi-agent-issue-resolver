import asyncio
from pathlib import Path

from miresolver import runtime
from miresolver.agents.types import ResolverOptions
from miresolver.models.settings import load_model_settings

SETTINGS = load_model_settings("""
tiers:
  embed: {provider: openai-compatible, base_url: "http://x/v1", model: e, api_key_env: K}
  screen: {provider: openai-compatible, base_url: "http://x/v1", model: s, api_key_env: K}
  investigate: {provider: openai-compatible, base_url: "http://x/v1", model: i, api_key_env: K}
  synthesize: {provider: openai-compatible, base_url: "http://x/v1", model: y, api_key_env: K}
""")


class RecordingQdrant:
    made = []

    def __init__(self, **kwargs):
        RecordingQdrant.made.append(kwargs)

    def close(self):
        pass


def test_the_qdrant_client_tolerates_a_slow_server(monkeypatch, tmp_path):
    # A five-second default turned brief server stalls into failed cases.
    monkeypatch.setattr(runtime, "QdrantClient", RecordingQdrant)

    async def go():
        async with runtime.open_runtime(
            SETTINGS, {"K": "k"}, qdrant_url="http://q:6333", collection="c",
            repo="o/r", local_repo=None, code_cache=Path(tmp_path),
            options=ResolverOptions(),
        ):
            pass

    asyncio.run(go())
    assert RecordingQdrant.made[-1]["url"] == "http://q:6333"
    assert RecordingQdrant.made[-1]["timeout"] >= 30

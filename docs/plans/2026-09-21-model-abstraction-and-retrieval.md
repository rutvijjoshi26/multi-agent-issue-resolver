# Model Abstraction and Retrieval Implementation Plan


**Goal:** Given the text of a new issue, return the past issues most likely to be relevant, retrieved from a Qdrant index by combined vector and keyword search, with embeddings produced by whichever model provider the user configures.

**Architecture:** A provider-agnostic model layer sits behind one `EmbeddingClient` protocol, with an OpenAI-compatible adapter covering most providers including self-hosted ones, plus native Anthropic and Gemini adapters. Which provider and model serve each tier is read from a config file, never hardcoded. Issue text from the Postgres corpus is embedded and written to Qdrant with the issue's payload, and search combines dense vector similarity with Qdrant's built-in keyword matching, because issue text is full of stack traces and symbol names that embeddings smear but exact matching nails.

**Tech Stack:** Python 3.12, qdrant-client, httpx, PyYAML, pytest, Docker Compose (Qdrant alongside the existing Postgres).

**Spec:** `docs/specs/2026-09-21-multi-agent-issue-resolver-design.md`

## Global Constraints

- Python 3.12 or newer. Type hints on all public functions.
- Bring your own model is a hard requirement, not a convenience. A user must be able to point any tier at a private internal model by editing config, with no code change and no redeploy.
- Provider and model per tier are read from config. No provider name or model id appears as a literal anywhere in `src/`, tests and example config excluded.
- No network calls in unit tests. Providers are exercised through recorded fixtures and `httpx.MockTransport`; Qdrant tests use an in-process client or skip when `TEST_QDRANT_URL` is unset, mirroring how `TEST_DATABASE_URL` gates the Postgres tests.
- Embedded text is symptom text: issue title, body, and early comments. The resolution, changed files, and base commit ride along as payload and are never embedded, so new symptoms match old symptoms rather than old solutions.
- Every retrieval result carries the issue number and the payload needed by later plans, never a bare score.

## What Already Exists (merged to master)

Read these before starting; do not re-derive or duplicate them.

- `src/miresolver/config.py`: `Config(github_token, database_url: str | None, target_repo)`, `ConfigError`, `read_dotenv(path=None) -> dict[str, str]`, `load_config(env, *, require_database_url=True) -> Config`.
- `src/miresolver/rules.py`: `CandidateIssue(number, title, body, labels, linked_pr_numbers, linkage_truncated=False)`, `CandidatePR(number, merged, title, changed_files, base_commit="", files_truncated=False)`, `source_files(paths) -> tuple[str, ...]`, `is_eligible(issue, prs) -> Eligibility`.
- `src/miresolver/store.py`: `create_schema(conn)`, `save_issues(conn, issues, prs)`, `load_issue_numbers(conn) -> list[int]`. Tables `issues`, `pull_requests`, `issue_pr_links`.
- `src/miresolver/manifest.py`: `EvalCase(issue_number, pr_number, base_commit, changed_files)`, `Manifest(dev, full, rejected, meta)`, `ManifestSizeError`.
- `src/miresolver/cli.py`: `main(argv=None) -> int`, subcommand `ingest`. New subcommands are added here.
- `eval/manifest.json`: the committed evaluation set.
- Tests run with `venv/bin/python3 -m pytest tests/ -q`. Currently 75 passed, 6 skipped.

---

### Task 1: Model provider config

**Files:**
- Create: `src/miresolver/models/__init__.py`
- Create: `src/miresolver/models/settings.py`
- Create: `config/models.example.yaml`
- Modify: `pyproject.toml` (add `pyyaml>=6.0` to dependencies)
- Test: `tests/models/test_settings.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `@dataclass(frozen=True) class TierSettings` with fields `provider: str`, `model: str`, `base_url: str | None`, `api_key_env: str | None`
  - `@dataclass(frozen=True) class ModelSettings` with field `tiers: Mapping[str, TierSettings]` and method `tier(name: str) -> TierSettings` raising `ModelConfigError` for an unknown tier
  - `class ModelConfigError(Exception)`
  - `load_model_settings(text: str) -> ModelSettings` parsing YAML from a string, so tests never touch the filesystem
  - `TIER_NAMES: tuple[str, ...] = ("embed", "screen", "investigate", "synthesize")`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_settings.py
import pytest

from miresolver.models.settings import (
    ModelConfigError,
    TierSettings,
    load_model_settings,
)

YAML = """
tiers:
  embed:
    provider: openai-compatible
    base_url: http://internal-llm:8000/v1
    model: some-embedding-model
    api_key_env: INTERNAL_LLM_KEY
  screen:
    provider: anthropic
    model: some-small-model
  investigate:
    provider: anthropic
    model: some-mid-model
  synthesize:
    provider: gemini
    model: some-large-model
"""


def test_loads_every_tier():
    settings = load_model_settings(YAML)
    assert settings.tier("embed") == TierSettings(
        provider="openai-compatible",
        model="some-embedding-model",
        base_url="http://internal-llm:8000/v1",
        api_key_env="INTERNAL_LLM_KEY",
    )


def test_optional_fields_default_to_none():
    settings = load_model_settings(YAML)
    screen = settings.tier("screen")
    assert screen.base_url is None
    assert screen.api_key_env is None


def test_unknown_tier_raises():
    settings = load_model_settings(YAML)
    with pytest.raises(ModelConfigError, match="unknown tier"):
        settings.tier("nonexistent")


def test_missing_tier_in_file_raises():
    with pytest.raises(ModelConfigError, match="investigate"):
        load_model_settings(
            "tiers:\n"
            "  embed: {provider: anthropic, model: m}\n"
            "  screen: {provider: anthropic, model: m}\n"
            "  synthesize: {provider: anthropic, model: m}\n"
        )


def test_missing_provider_raises():
    with pytest.raises(ModelConfigError, match="provider"):
        load_model_settings("tiers:\n  embed: {model: m}\n")


def test_missing_model_raises():
    with pytest.raises(ModelConfigError, match="model"):
        load_model_settings("tiers:\n  embed: {provider: anthropic}\n")


def test_empty_document_raises():
    with pytest.raises(ModelConfigError, match="tiers"):
        load_model_settings("")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python3 -m pytest tests/models/test_settings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'miresolver.models'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/miresolver/models/__init__.py
```

```python
# src/miresolver/models/settings.py
from collections.abc import Mapping
from dataclasses import dataclass

import yaml

TIER_NAMES: tuple[str, ...] = ("embed", "screen", "investigate", "synthesize")


class ModelConfigError(Exception):
    """Raised when model configuration is absent or malformed."""


@dataclass(frozen=True)
class TierSettings:
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class ModelSettings:
    tiers: Mapping[str, TierSettings]

    def tier(self, name: str) -> TierSettings:
        try:
            return self.tiers[name]
        except KeyError as exc:
            raise ModelConfigError(f"unknown tier: {name}") from exc


def _tier_from_mapping(name: str, raw: object) -> TierSettings:
    if not isinstance(raw, Mapping):
        raise ModelConfigError(f"tier {name} must be a mapping")
    for required in ("provider", "model"):
        if not raw.get(required):
            raise ModelConfigError(f"tier {name} is missing {required}")
    return TierSettings(
        provider=str(raw["provider"]),
        model=str(raw["model"]),
        base_url=raw.get("base_url"),
        api_key_env=raw.get("api_key_env"),
    )


def load_model_settings(text: str) -> ModelSettings:
    document = yaml.safe_load(text) or {}
    raw_tiers = document.get("tiers")
    if not isinstance(raw_tiers, Mapping) or not raw_tiers:
        raise ModelConfigError("config must define a non-empty tiers mapping")

    tiers = {
        name: _tier_from_mapping(name, raw)
        for name, raw in raw_tiers.items()
    }
    missing = [name for name in TIER_NAMES if name not in tiers]
    if missing:
        raise ModelConfigError(f"config is missing tiers: {', '.join(missing)}")
    return ModelSettings(tiers=tiers)
```

```yaml
# config/models.example.yaml
# Point any tier at any provider. Nothing here is baked into the code.
#
# provider: openai-compatible works with vLLM, Ollama, LM Studio, Together,
# Groq, Azure OpenAI, and most internal company gateways. Set base_url to
# your endpoint and api_key_env to the environment variable holding the key.
#
# provider: anthropic and provider: gemini talk to those APIs natively.
#
# Replace every model id below with one your provider actually serves.
tiers:
  embed:
    provider: openai-compatible
    base_url: http://localhost:11434/v1
    model: REPLACE_WITH_YOUR_EMBEDDING_MODEL
    api_key_env: LOCAL_LLM_API_KEY
  screen:
    provider: openai-compatible
    base_url: http://localhost:11434/v1
    model: REPLACE_WITH_A_SMALL_FAST_MODEL
    api_key_env: LOCAL_LLM_API_KEY
  investigate:
    provider: openai-compatible
    base_url: http://localhost:11434/v1
    model: REPLACE_WITH_A_MID_TIER_MODEL
    api_key_env: LOCAL_LLM_API_KEY
  synthesize:
    provider: openai-compatible
    base_url: http://localhost:11434/v1
    model: REPLACE_WITH_YOUR_STRONGEST_MODEL
    api_key_env: LOCAL_LLM_API_KEY
```

Add `"pyyaml>=6.0"` to the `dependencies` list in `pyproject.toml`.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pip install -e ".[dev]" && venv/bin/python3 -m pytest tests/models/test_settings.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/miresolver/models config/models.example.yaml tests/models pyproject.toml
git commit -m "feat: per-tier model provider configuration"
```

---

### Task 2: Embedding protocol and the OpenAI-compatible adapter

**Files:**
- Create: `src/miresolver/models/embedding.py`
- Create: `tests/models/test_openai_compatible.py`

**Interfaces:**
- Consumes: `TierSettings` from Task 1
- Produces:
  - `class EmbeddingClient(Protocol)` with `async def embed(self, texts: Sequence[str]) -> list[list[float]]` and `async def aclose(self) -> None`
  - `class EmbeddingError(Exception)`
  - `class OpenAICompatibleEmbedder` with `__init__(self, settings: TierSettings, api_key: str, *, transport: httpx.AsyncBaseTransport | None = None)`, implementing the protocol
  - `parse_embedding_response(payload: dict) -> list[list[float]]` as a pure function, ordering vectors by the response's `index` field

Ordering matters: the OpenAI embeddings API does not guarantee response order matches request order, and a silently transposed batch would corrupt every vector in the index while every test that checks only shapes would still pass.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_openai_compatible.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python3 -m pytest tests/models/test_openai_compatible.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'miresolver.models.embedding'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/miresolver/models/embedding.py
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
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embedding request failed: {exc}") from exc
        return parse_embedding_response(response.json())

    async def aclose(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python3 -m pytest tests/models/test_openai_compatible.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/miresolver/models/embedding.py tests/models/test_openai_compatible.py
git commit -m "feat: embedding protocol and OpenAI-compatible adapter"
```

---

### Task 3: Native Anthropic and Gemini adapters, and the factory

**Files:**
- Create: `src/miresolver/models/anthropic.py`
- Create: `src/miresolver/models/gemini.py`
- Create: `src/miresolver/models/factory.py`
- Test: `tests/models/test_factory.py`

**Interfaces:**
- Consumes: `TierSettings`, `ModelSettings`, `ModelConfigError` from Task 1; `EmbeddingClient`, `EmbeddingError` from Task 2
- Produces:
  - `class AnthropicEmbedder` and `class GeminiEmbedder`, both raising `EmbeddingError` from `embed` with a message naming the provider, because neither vendor's embedding story is stable enough to hardcode; users route `embed` at an OpenAI-compatible endpoint
  - `build_embedder(settings: ModelSettings, env: Mapping[str, str], *, transport=None) -> EmbeddingClient` reading the `embed` tier and dispatching on `provider`
  - `PROVIDERS: frozenset[str]` naming every supported provider string

The two native adapters are deliberately thin in this plan. The chat-completion side of Anthropic and Gemini arrives in plan 3, where agents need it; embedding is the only model capability plan 2 uses, and shipping stubs that raise clearly beats shipping guesses that silently produce wrong vectors.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_factory.py
import asyncio

import pytest

from miresolver.models.embedding import EmbeddingError, OpenAICompatibleEmbedder
from miresolver.models.factory import PROVIDERS, build_embedder
from miresolver.models.settings import ModelConfigError, load_model_settings

def settings_for(provider: str, *, base_url: str | None = None) -> str:
    url = f"\n    base_url: {base_url}" if base_url else ""
    return (
        "tiers:\n"
        f"  embed:\n    provider: {provider}\n    model: m{url}\n"
        "    api_key_env: TEST_KEY\n"
        "  screen: {provider: anthropic, model: m, api_key_env: TEST_KEY}\n"
        "  investigate: {provider: anthropic, model: m, api_key_env: TEST_KEY}\n"
        "  synthesize: {provider: anthropic, model: m, api_key_env: TEST_KEY}\n"
    )


ENV = {"TEST_KEY": "secret"}


def test_builds_an_openai_compatible_embedder():
    settings = load_model_settings(
        settings_for("openai-compatible", base_url="http://host/v1")
    )
    embedder = build_embedder(settings, ENV)
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    asyncio.run(embedder.aclose())


def test_anthropic_embedder_refuses_clearly_rather_than_guessing():
    settings = load_model_settings(settings_for("anthropic"))
    embedder = build_embedder(settings, ENV)
    with pytest.raises(EmbeddingError, match="anthropic"):
        asyncio.run(embedder.embed(["hello"]))


def test_gemini_embedder_refuses_clearly_rather_than_guessing():
    settings = load_model_settings(settings_for("gemini"))
    embedder = build_embedder(settings, ENV)
    with pytest.raises(EmbeddingError, match="gemini"):
        asyncio.run(embedder.embed(["hello"]))


def test_unknown_provider_raises():
    settings = load_model_settings(settings_for("not-a-provider"))
    with pytest.raises(ModelConfigError, match="not-a-provider"):
        build_embedder(settings, ENV)


def test_missing_api_key_in_environment_raises():
    settings = load_model_settings(
        settings_for("openai-compatible", base_url="http://host/v1")
    )
    with pytest.raises(ModelConfigError, match="TEST_KEY"):
        build_embedder(settings, {})


def test_provider_set_is_what_the_factory_dispatches_on():
    assert PROVIDERS == frozenset({"openai-compatible", "anthropic", "gemini"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python3 -m pytest tests/models/test_factory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'miresolver.models.factory'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/miresolver/models/anthropic.py
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
```

```python
# src/miresolver/models/gemini.py
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
```

```python
# src/miresolver/models/factory.py
from collections.abc import Mapping

import httpx

from miresolver.models.anthropic import AnthropicEmbedder
from miresolver.models.embedding import EmbeddingClient, OpenAICompatibleEmbedder
from miresolver.models.gemini import GeminiEmbedder
from miresolver.models.settings import ModelConfigError, ModelSettings, TierSettings

PROVIDERS: frozenset[str] = frozenset(
    {"openai-compatible", "anthropic", "gemini"}
)


def _api_key(tier: TierSettings, env: Mapping[str, str]) -> str:
    if not tier.api_key_env:
        raise ModelConfigError(
            f"tier using provider {tier.provider} is missing api_key_env"
        )
    key = env.get(tier.api_key_env)
    if not key:
        raise ModelConfigError(
            f"environment variable {tier.api_key_env} is not set"
        )
    return key


def build_embedder(
    settings: ModelSettings,
    env: Mapping[str, str],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> EmbeddingClient:
    tier = settings.tier("embed")
    if tier.provider not in PROVIDERS:
        raise ModelConfigError(f"unknown provider: {tier.provider}")
    key = _api_key(tier, env)
    if tier.provider == "openai-compatible":
        return OpenAICompatibleEmbedder(tier, key, transport=transport)
    if tier.provider == "anthropic":
        return AnthropicEmbedder(tier, key)
    return GeminiEmbedder(tier, key)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python3 -m pytest tests/models/test_factory.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/miresolver/models tests/models/test_factory.py
git commit -m "feat: provider factory with native adapter placeholders"
```

---

### Task 4: What gets embedded, and what rides along as payload

**Files:**
- Create: `src/miresolver/retrieval/__init__.py`
- Create: `src/miresolver/retrieval/documents.py`
- Test: `tests/retrieval/test_documents.py`

**Interfaces:**
- Consumes: `CandidateIssue` from `miresolver.rules`
- Produces:
  - `@dataclass(frozen=True) class IssueDocument` with fields `issue_number: int`, `embed_text: str`, `payload: Mapping[str, object]`
  - `build_document(issue: CandidateIssue, *, pr_number: int | None = None, base_commit: str = "", changed_files: Sequence[str] = ()) -> IssueDocument`
  - `MAX_EMBED_CHARS: int = 8000`

The spec is explicit that symptom text is embedded and the resolution is not. This task is where that rule is enforced, so the test asserts the fix does not leak into `embed_text`.

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_documents.py
from miresolver.retrieval.documents import MAX_EMBED_CHARS, build_document
from miresolver.rules import CandidateIssue

ISSUE = CandidateIssue(
    number=4821,
    title="crash when config file is empty",
    body="Opening an empty settings file crashes the window.",
    labels=("bug",),
    linked_pr_numbers=(4830,),
)


def test_embed_text_contains_title_and_body():
    doc = build_document(ISSUE)
    assert "crash when config file is empty" in doc.embed_text
    assert "Opening an empty settings file" in doc.embed_text


def test_embed_text_excludes_the_resolution():
    doc = build_document(
        ISSUE,
        pr_number=4830,
        base_commit="abc123",
        changed_files=("src/config/parser.ts",),
    )
    assert "parser.ts" not in doc.embed_text
    assert "abc123" not in doc.embed_text
    assert "4830" not in doc.embed_text


def test_resolution_travels_in_the_payload():
    doc = build_document(
        ISSUE,
        pr_number=4830,
        base_commit="abc123",
        changed_files=("src/config/parser.ts",),
    )
    assert doc.payload["pr_number"] == 4830
    assert doc.payload["base_commit"] == "abc123"
    assert doc.payload["changed_files"] == ["src/config/parser.ts"]


def test_payload_carries_the_symptom_text_for_display():
    doc = build_document(ISSUE)
    assert doc.payload["title"] == "crash when config file is empty"
    assert doc.payload["issue_number"] == 4821


def test_embed_text_is_truncated_at_the_limit():
    long_issue = CandidateIssue(
        number=1,
        title="t",
        body="x" * (MAX_EMBED_CHARS * 2),
        labels=("bug",),
        linked_pr_numbers=(),
    )
    doc = build_document(long_issue)
    assert len(doc.embed_text) == MAX_EMBED_CHARS


def test_document_number_matches_the_issue():
    assert build_document(ISSUE).issue_number == 4821
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python3 -m pytest tests/retrieval/test_documents.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'miresolver.retrieval'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/miresolver/retrieval/__init__.py
```

```python
# src/miresolver/retrieval/documents.py
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from miresolver.rules import CandidateIssue

MAX_EMBED_CHARS = 8000


@dataclass(frozen=True)
class IssueDocument:
    issue_number: int
    embed_text: str
    payload: Mapping[str, object]


def build_document(
    issue: CandidateIssue,
    *,
    pr_number: int | None = None,
    base_commit: str = "",
    changed_files: Sequence[str] = (),
) -> IssueDocument:
    """Turn an issue into one indexable document.

    Only symptom text is embedded. The resolution rides along as payload so
    that a new symptom matches old symptoms rather than old solutions.
    """
    embed_text = f"{issue.title}\n\n{issue.body}".strip()[:MAX_EMBED_CHARS]
    payload: dict[str, object] = {
        "issue_number": issue.number,
        "title": issue.title,
        "body": issue.body[:MAX_EMBED_CHARS],
        "labels": list(issue.labels),
        "pr_number": pr_number,
        "base_commit": base_commit,
        "changed_files": list(changed_files),
    }
    return IssueDocument(
        issue_number=issue.number, embed_text=embed_text, payload=payload
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python3 -m pytest tests/retrieval/test_documents.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/miresolver/retrieval tests/retrieval
git commit -m "feat: issue documents separating symptom text from resolution"
```

---

### Task 5: Qdrant index with hybrid search

**Files:**
- Create: `src/miresolver/retrieval/index.py`
- Modify: `docker-compose.yml` (add the Qdrant service)
- Modify: `pyproject.toml` (add `qdrant-client>=1.12`)
- Test: `tests/retrieval/test_index.py`
- Modify: `tests/conftest.py` (add the `qdrant` fixture)

**Interfaces:**
- Consumes: `IssueDocument` from Task 4
- Produces:
  - `@dataclass(frozen=True) class SearchHit` with fields `issue_number: int`, `score: float`, `payload: Mapping[str, object]`
  - `class IssueIndex` with:
    - `__init__(self, client: QdrantClient, *, collection: str = "issues")`
    - `create(self, vector_size: int) -> None`, idempotent
    - `upsert(self, documents: Sequence[IssueDocument], vectors: Sequence[Sequence[float]]) -> None` raising `ValueError` when the counts differ
    - `search_dense(self, vector: Sequence[float], *, limit: int) -> list[SearchHit]`
    - `search_keyword(self, text: str, *, limit: int) -> list[SearchHit]`
    - `count(self) -> int`
  - `fuse(dense: Sequence[SearchHit], keyword: Sequence[SearchHit], *, limit: int, k: int = 60) -> list[SearchHit]` implementing reciprocal rank fusion

Fusion is a pure function so the ranking logic is testable without a running Qdrant, which is the part most worth testing. Reciprocal rank fusion is used rather than score averaging because dense cosine scores and keyword scores are not on a comparable scale, and averaging them lets whichever has the wider range dominate.

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_index.py
import pytest

from miresolver.retrieval.documents import build_document
from miresolver.retrieval.index import IssueIndex, SearchHit, fuse
from miresolver.rules import CandidateIssue


def hit(number: int, score: float) -> SearchHit:
    return SearchHit(issue_number=number, score=score, payload={})


def test_fuse_ranks_an_item_found_by_both_above_one_found_by_either():
    dense = [hit(1, 0.9), hit(2, 0.8)]
    keyword = [hit(3, 40.0), hit(2, 30.0)]
    fused = fuse(dense, keyword, limit=3)
    assert fused[0].issue_number == 2


def test_fuse_is_insensitive_to_score_scale():
    dense = [hit(1, 0.9), hit(2, 0.89)]
    keyword = [hit(2, 9999.0), hit(1, 1.0)]
    fused = fuse(dense, keyword, limit=2)
    assert {h.issue_number for h in fused} == {1, 2}
    assert fused[0].issue_number == 1


def test_fuse_respects_the_limit():
    dense = [hit(i, 1.0) for i in range(10)]
    assert len(fuse(dense, [], limit=3)) == 3


def test_fuse_deduplicates_by_issue_number():
    fused = fuse([hit(1, 0.9)], [hit(1, 50.0)], limit=5)
    assert len(fused) == 1


def test_fuse_keeps_the_payload_from_whichever_side_had_it():
    dense = [SearchHit(issue_number=1, score=0.9, payload={"title": "t"})]
    fused = fuse(dense, [hit(1, 5.0)], limit=1)
    assert fused[0].payload == {"title": "t"}


def test_upsert_rejects_mismatched_counts(qdrant):
    index = IssueIndex(qdrant, collection="test_mismatch")
    index.create(vector_size=3)
    issue = CandidateIssue(
        number=1, title="t", body="b", labels=("bug",), linked_pr_numbers=()
    )
    with pytest.raises(ValueError, match="one vector per document"):
        index.upsert([build_document(issue)], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


def test_round_trip_dense_search_returns_the_nearest_document(qdrant):
    index = IssueIndex(qdrant, collection="test_dense")
    index.create(vector_size=3)
    issues = [
        CandidateIssue(
            number=n, title=f"issue {n}", body="b", labels=("bug",),
            linked_pr_numbers=(),
        )
        for n in (1, 2)
    ]
    docs = [build_document(i) for i in issues]
    index.upsert(docs, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert index.count() == 2

    hits = index.search_dense([0.99, 0.01, 0.0], limit=1)
    assert [h.issue_number for h in hits] == [1]
    assert hits[0].payload["title"] == "issue 1"


def test_keyword_search_finds_an_exact_symbol(qdrant):
    index = IssueIndex(qdrant, collection="test_keyword")
    index.create(vector_size=3)
    issues = [
        CandidateIssue(
            number=1, title="ErrInvalidConfigFormat on startup", body="b",
            labels=("bug",), linked_pr_numbers=(),
        ),
        CandidateIssue(
            number=2, title="window flickers on resize", body="b",
            labels=("bug",), linked_pr_numbers=(),
        ),
    ]
    docs = [build_document(i) for i in issues]
    index.upsert(docs, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    hits = index.search_keyword("ErrInvalidConfigFormat", limit=5)
    assert [h.issue_number for h in hits] == [1]


def test_create_is_idempotent(qdrant):
    index = IssueIndex(qdrant, collection="test_idempotent")
    index.create(vector_size=3)
    index.create(vector_size=3)
    assert index.count() == 0
```

Add to `tests/conftest.py`:

```python
@pytest.fixture
def qdrant():
    """In-memory Qdrant, so index tests need no running service."""
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    yield client
    client.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python3 -m pytest tests/retrieval/test_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'miresolver.retrieval.index'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/miresolver/retrieval/index.py
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from miresolver.retrieval.documents import IssueDocument

DEFAULT_COLLECTION = "issues"


@dataclass(frozen=True)
class SearchHit:
    issue_number: int
    score: float
    payload: Mapping[str, object]


def fuse(
    dense: Sequence[SearchHit],
    keyword: Sequence[SearchHit],
    *,
    limit: int,
    k: int = 60,
) -> list[SearchHit]:
    """Reciprocal rank fusion.

    Dense cosine scores and keyword scores are not on a comparable scale, so
    ranks are combined rather than the scores themselves.
    """
    scores: dict[int, float] = {}
    payloads: dict[int, Mapping[str, object]] = {}
    for ranking in (dense, keyword):
        for rank, hit in enumerate(ranking):
            scores[hit.issue_number] = scores.get(hit.issue_number, 0.0) + 1.0 / (
                k + rank + 1
            )
            if hit.payload and hit.issue_number not in payloads:
                payloads[hit.issue_number] = hit.payload
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [
        SearchHit(
            issue_number=number, score=score, payload=payloads.get(number, {})
        )
        for number, score in ordered[:limit]
    ]


class IssueIndex:
    def __init__(
        self, client: QdrantClient, *, collection: str = DEFAULT_COLLECTION
    ) -> None:
        self._client = client
        self._collection = collection

    def create(self, vector_size: int) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="embed_text",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
            ),
        )

    def upsert(
        self,
        documents: Sequence[IssueDocument],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(documents) != len(vectors):
            raise ValueError(
                f"one vector per document required: {len(documents)} documents, "
                f"{len(vectors)} vectors"
            )
        points = [
            models.PointStruct(
                id=doc.issue_number,
                vector=list(vector),
                payload={**doc.payload, "embed_text": doc.embed_text},
            )
            for doc, vector in zip(documents, vectors, strict=True)
        ]
        if points:
            self._client.upsert(collection_name=self._collection, points=points)

    def _to_hits(self, points: Sequence[object]) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            payload.pop("embed_text", None)
            hits.append(
                SearchHit(
                    issue_number=int(payload.get("issue_number", getattr(point, "id"))),
                    score=float(getattr(point, "score", 0.0)),
                    payload=payload,
                )
            )
        return hits

    def search_dense(
        self, vector: Sequence[float], *, limit: int
    ) -> list[SearchHit]:
        result = self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            limit=limit,
            with_payload=True,
        )
        return self._to_hits(result.points)

    def search_keyword(self, text: str, *, limit: int) -> list[SearchHit]:
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="embed_text", match=models.MatchText(text=text)
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        )
        return self._to_hits(points)

    def count(self) -> int:
        return self._client.count(
            collection_name=self._collection, exact=True
        ).count
```

Add the Qdrant service to `docker-compose.yml`:

```yaml
  qdrant:
    image: qdrant/qdrant:v1.12.4
    ports: ["6333:6333"]
    volumes: ["qdrantdata:/qdrant/storage"]
```

and add `qdrantdata:` under the existing `volumes:` key.

Add `"qdrant-client>=1.12"` to `dependencies` in `pyproject.toml`.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pip install -e ".[dev]" && venv/bin/python3 -m pytest tests/retrieval/test_index.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/miresolver/retrieval/index.py tests/retrieval/test_index.py tests/conftest.py docker-compose.yml pyproject.toml
git commit -m "feat: Qdrant index with dense, keyword, and fused search"
```

---

### Task 6: Index-building pipeline

**Files:**
- Create: `src/miresolver/retrieval/pipeline.py`
- Test: `tests/retrieval/test_pipeline.py`

**Interfaces:**
- Consumes: `EmbeddingClient` (Task 2), `IssueDocument` and `build_document` (Task 4), `IssueIndex` (Task 5)
- Produces:
  - `async def build_index(documents: Sequence[IssueDocument], embedder: EmbeddingClient, index: IssueIndex, *, batch_size: int = 64, vector_size: int | None = None) -> int` returning the number of documents indexed
  - `async def search(query: str, embedder: EmbeddingClient, index: IssueIndex, *, limit: int = 10) -> list[SearchHit]` running dense and keyword search and fusing them

`build_index` learns the vector size from the first batch rather than requiring it up front, because the size is a property of whichever model the user configured and hardcoding it would break bring-your-own-model.

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_pipeline.py
import asyncio

import pytest

from miresolver.retrieval.documents import build_document
from miresolver.retrieval.index import IssueIndex
from miresolver.retrieval.pipeline import build_index, search
from miresolver.rules import CandidateIssue


class FakeEmbedder:
    """Maps text to a deterministic 3-vector; records call batches."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def embed(self, texts):
        self.batches.append(list(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    async def aclose(self) -> None:
        return None


def issues(count: int) -> list[CandidateIssue]:
    return [
        CandidateIssue(
            number=n, title=f"issue {n}", body="b" * n, labels=("bug",),
            linked_pr_numbers=(),
        )
        for n in range(1, count + 1)
    ]


def test_build_index_indexes_every_document(qdrant):
    index = IssueIndex(qdrant, collection="test_pipeline_all")
    docs = [build_document(i) for i in issues(5)]
    indexed = asyncio.run(build_index(docs, FakeEmbedder(), index))
    assert indexed == 5
    assert index.count() == 5


def test_build_index_batches_requests(qdrant):
    index = IssueIndex(qdrant, collection="test_pipeline_batch")
    embedder = FakeEmbedder()
    docs = [build_document(i) for i in issues(5)]
    asyncio.run(build_index(docs, embedder, index, batch_size=2))
    assert [len(b) for b in embedder.batches] == [2, 2, 1]


def test_build_index_handles_an_empty_corpus(qdrant):
    index = IssueIndex(qdrant, collection="test_pipeline_empty")
    embedder = FakeEmbedder()
    assert asyncio.run(build_index([], embedder, index)) == 0
    assert embedder.batches == []


def test_build_index_infers_vector_size_from_the_model(qdrant):
    index = IssueIndex(qdrant, collection="test_pipeline_size")
    docs = [build_document(i) for i in issues(1)]
    asyncio.run(build_index(docs, FakeEmbedder(), index))
    assert index.count() == 1


def test_search_returns_fused_hits(qdrant):
    index = IssueIndex(qdrant, collection="test_pipeline_search")
    docs = [build_document(i) for i in issues(3)]
    embedder = FakeEmbedder()
    asyncio.run(build_index(docs, embedder, index))
    hits = asyncio.run(search("issue 2", embedder, index, limit=3))
    assert hits
    assert all(h.issue_number in {1, 2, 3} for h in hits)


def test_search_on_an_empty_index_returns_nothing(qdrant):
    index = IssueIndex(qdrant, collection="test_pipeline_searchempty")
    index.create(vector_size=3)
    assert asyncio.run(search("anything", FakeEmbedder(), index)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python3 -m pytest tests/retrieval/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'miresolver.retrieval.pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/miresolver/retrieval/pipeline.py
from collections.abc import Sequence

from miresolver.models.embedding import EmbeddingClient
from miresolver.retrieval.documents import IssueDocument
from miresolver.retrieval.index import IssueIndex, SearchHit, fuse


async def build_index(
    documents: Sequence[IssueDocument],
    embedder: EmbeddingClient,
    index: IssueIndex,
    *,
    batch_size: int = 64,
    vector_size: int | None = None,
) -> int:
    """Embed every document and write it to the index.

    The vector size comes from the first batch rather than a constant,
    because it is a property of whichever model the user configured.
    """
    if not documents:
        return 0

    indexed = 0
    created = False
    for start in range(0, len(documents), batch_size):
        batch = list(documents[start : start + batch_size])
        vectors = await embedder.embed([doc.embed_text for doc in batch])
        if not created:
            size = vector_size if vector_size is not None else len(vectors[0])
            index.create(vector_size=size)
            created = True
        index.upsert(batch, vectors)
        indexed += len(batch)
    return indexed


async def search(
    query: str,
    embedder: EmbeddingClient,
    index: IssueIndex,
    *,
    limit: int = 10,
) -> list[SearchHit]:
    """Dense and keyword search over the same query, fused by rank."""
    vectors = await embedder.embed([query])
    dense = index.search_dense(vectors[0], limit=limit)
    keyword = index.search_keyword(query, limit=limit)
    return fuse(dense, keyword, limit=limit)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python3 -m pytest tests/retrieval/test_pipeline.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/miresolver/retrieval/pipeline.py tests/retrieval/test_pipeline.py
git commit -m "feat: batched index building and fused search"
```

---

### Task 7: `index` and `search` CLI subcommands

**Files:**
- Modify: `src/miresolver/cli.py`
- Create: `src/miresolver/retrieval/loader.py`
- Test: `tests/retrieval/test_loader.py`
- Test: `tests/test_cli_search.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything from Tasks 1 to 6, plus `psycopg` and the existing `issues` / `pull_requests` / `issue_pr_links` tables
- Produces:
  - `load_documents(conn) -> list[IssueDocument]` in `loader.py`, joining issues to their single merged PR so the payload carries `pr_number`, `base_commit`, and `changed_files`
  - CLI subcommand `index` with flags `--models config/models.yaml`, `--qdrant-url`, `--collection`, `--batch-size`
  - CLI subcommand `search` with a positional `query` and flags `--models`, `--qdrant-url`, `--collection`, `--limit`
  - Both exit non-zero with a clear message on `ModelConfigError`, `EmbeddingError`, or `ConfigError`, never a traceback

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_loader.py
from miresolver.retrieval.loader import load_documents
from miresolver.rules import CandidateIssue, CandidatePR
from miresolver.store import create_schema, save_issues

ISSUE = CandidateIssue(
    number=4821,
    title="crash when config file is empty",
    body="it crashes",
    labels=("bug",),
    linked_pr_numbers=(4830,),
)
PR = CandidatePR(
    number=4830,
    merged=True,
    title="fix parser crash",
    changed_files=("src/config/parser.ts",),
    base_commit="abc123",
)


def test_load_documents_joins_the_resolution_into_the_payload(conn):
    create_schema(conn)
    save_issues(conn, [ISSUE], {4830: PR})
    docs = load_documents(conn)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.issue_number == 4821
    assert doc.payload["pr_number"] == 4830
    assert doc.payload["base_commit"] == "abc123"
    assert doc.payload["changed_files"] == ["src/config/parser.ts"]


def test_load_documents_does_not_embed_the_resolution(conn):
    create_schema(conn)
    save_issues(conn, [ISSUE], {4830: PR})
    doc = load_documents(conn)[0]
    assert "parser.ts" not in doc.embed_text
    assert "abc123" not in doc.embed_text


def test_load_documents_returns_empty_for_an_empty_corpus(conn):
    create_schema(conn)
    assert load_documents(conn) == []
```

```python
# tests/test_cli_search.py
from miresolver.cli import main

MODELS_YAML = """
tiers:
  embed: {provider: openai-compatible, base_url: "http://x/v1", model: m, api_key_env: K}
  screen: {provider: anthropic, model: m, api_key_env: K}
  investigate: {provider: anthropic, model: m, api_key_env: K}
  synthesize: {provider: anthropic, model: m, api_key_env: K}
"""


def test_search_reports_a_missing_models_file_cleanly(tmp_path, capsys):
    exit_code = main(["search", "anything", "--models", str(tmp_path / "nope.yaml")])
    assert exit_code == 1
    assert "nope.yaml" in capsys.readouterr().err


def test_search_reports_a_missing_api_key_cleanly(tmp_path, capsys, monkeypatch):
    models = tmp_path / "models.yaml"
    models.write_text(MODELS_YAML)
    monkeypatch.delenv("K", raising=False)
    exit_code = main(["search", "anything", "--models", str(models)])
    assert exit_code == 1
    assert "K" in capsys.readouterr().err


def test_index_reports_a_missing_models_file_cleanly(tmp_path, capsys):
    exit_code = main(["index", "--models", str(tmp_path / "nope.yaml")])
    assert exit_code == 1
    assert "nope.yaml" in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python3 -m pytest tests/retrieval/test_loader.py tests/test_cli_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'miresolver.retrieval.loader'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/miresolver/retrieval/loader.py
import psycopg

from miresolver.retrieval.documents import IssueDocument, build_document
from miresolver.rules import CandidateIssue

QUERY = """
SELECT i.number, i.title, i.body, i.labels,
       p.number, p.base_commit, p.changed_files
FROM issues i
LEFT JOIN issue_pr_links l ON l.issue_number = i.number
LEFT JOIN pull_requests p ON p.number = l.pr_number AND p.merged
ORDER BY i.number
"""


def load_documents(conn: psycopg.Connection) -> list[IssueDocument]:
    """Read the corpus and join each issue to its merged fixing PR."""
    with conn.cursor() as cur:
        cur.execute(QUERY)
        rows = cur.fetchall()

    documents: list[IssueDocument] = []
    for number, title, body, labels, pr_number, base_commit, changed in rows:
        issue = CandidateIssue(
            number=number,
            title=title,
            body=body,
            labels=tuple(labels or ()),
            linked_pr_numbers=(pr_number,) if pr_number else (),
        )
        documents.append(
            build_document(
                issue,
                pr_number=pr_number,
                base_commit=base_commit or "",
                changed_files=tuple(changed or ()),
            )
        )
    return documents
```

In `src/miresolver/cli.py`, add these imports at the top:

```python
from qdrant_client import QdrantClient

from miresolver.models.embedding import EmbeddingError
from miresolver.models.factory import build_embedder
from miresolver.models.settings import ModelConfigError, load_model_settings
from miresolver.retrieval.index import IssueIndex
from miresolver.retrieval.loader import load_documents
from miresolver.retrieval.pipeline import build_index, search
```

Add these helpers and command functions to `cli.py`:

```python
DEFAULT_MODELS_PATH = Path("config/models.yaml")
DEFAULT_QDRANT_URL = "http://localhost:6333"


def _load_models(path: Path):
    try:
        return load_model_settings(path.read_text())
    except OSError as exc:
        raise ModelConfigError(f"cannot read model config {path}: {exc}") from exc


def _run_index(args: argparse.Namespace) -> int:
    settings = _load_models(args.models)
    embedder = build_embedder(settings, os.environ)
    config = load_config({**read_dotenv(), **os.environ})
    client = QdrantClient(url=args.qdrant_url)
    try:
        with psycopg.connect(config.database_url, autocommit=True) as conn:
            documents = load_documents(conn)
        index = IssueIndex(client, collection=args.collection)
        indexed = asyncio.run(
            build_index(documents, embedder, index, batch_size=args.batch_size)
        )
    finally:
        asyncio.run(embedder.aclose())
        client.close()
    print(f"indexed {indexed} issues into {args.collection}")
    return 0


def _run_search(args: argparse.Namespace) -> int:
    settings = _load_models(args.models)
    embedder = build_embedder(settings, os.environ)
    client = QdrantClient(url=args.qdrant_url)
    try:
        index = IssueIndex(client, collection=args.collection)
        hits = asyncio.run(search(args.query, embedder, index, limit=args.limit))
    finally:
        asyncio.run(embedder.aclose())
        client.close()

    if not hits:
        print("no similar issues found")
        return 0
    for rank, hit in enumerate(hits, start=1):
        title = hit.payload.get("title", "")
        pr = hit.payload.get("pr_number")
        print(f"{rank}. #{hit.issue_number} {title}")
        print(f"   score {hit.score:.4f}   fixed by PR {pr}")
    return 0
```

Register both subcommands inside `main`, next to the existing `ingest` parser:

```python
    index_cmd = sub.add_parser("index", help="embed the corpus into Qdrant")
    index_cmd.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    index_cmd.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    index_cmd.add_argument("--collection", default="issues")
    index_cmd.add_argument("--batch-size", type=int, default=64)

    search_cmd = sub.add_parser("search", help="find similar past issues")
    search_cmd.add_argument("query")
    search_cmd.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    search_cmd.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    search_cmd.add_argument("--collection", default="issues")
    search_cmd.add_argument("--limit", type=int, default=10)
```

and dispatch them where `ingest` is dispatched, wrapping both in the error handler:

```python
    if args.command in ("index", "search"):
        try:
            return _run_index(args) if args.command == "index" else _run_search(args)
        except (ModelConfigError, EmbeddingError, ConfigError) as exc:
            print(f"{args.command} failed: {exc}", file=sys.stderr)
            return 1
```

Add a Retrieval section to `README.md`:

````markdown
## Retrieval

Bring your own model. Copy `config/models.example.yaml` to `config/models.yaml`
and point each tier at whatever you run, including a private internal endpoint.
Any service speaking the OpenAI API works through `provider:
openai-compatible`; set `base_url` and the environment variable named by
`api_key_env`. Changing provider or model is a config edit, never a code
change.

```bash
docker compose up -d postgres qdrant
cp config/models.example.yaml config/models.yaml   # then edit it
miresolver index
miresolver search "kubelet crashes when the config file is empty"
```

Search runs dense vector similarity and keyword matching over the same query
and fuses the two rankings. Keyword matching matters here because issue text is
full of stack traces and symbol names, which embeddings blur and exact matching
catches.

Only symptom text is embedded: the issue title and body. The fixing PR, the
files it changed, and the commit to index against travel alongside as payload,
so a new symptom is matched against old symptoms rather than old solutions.
````

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python3 -m pytest tests/ -q`
Expected: all pass, with the Postgres-gated tests skipping when `TEST_DATABASE_URL` is unset

- [ ] **Step 5: Commit**

```bash
git add src/miresolver/cli.py src/miresolver/retrieval/loader.py tests/retrieval/test_loader.py tests/test_cli_search.py README.md
git commit -m "feat: index and search CLI subcommands"
```

---

## Done when

- `venv/bin/python3 -m pytest tests/ -q` passes, with database tests skipping cleanly when `TEST_DATABASE_URL` is unset
- `grep -rE "gpt-|claude-|gemini-|text-embedding" src/` returns nothing: no provider's model id is hardcoded anywhere
- `miresolver index` embeds the corpus into Qdrant and reports the count
- `miresolver search "<text>"` returns ranked past issues with their fixing PR
- Pointing the embed tier at a different `base_url` and model requires editing only `config/models.yaml`

## What the next plan consumes from this one

- `EmbeddingClient` protocol and `build_embedder`, extended there with chat completion for the agent tiers
- `ModelSettings` and the per-tier config, already carrying `screen`, `investigate`, and `synthesize`
- `search(query, embedder, index, limit)` as the candidate generator feeding the screening pass
- `SearchHit.payload`, carrying the `pr_number`, `base_commit`, and `changed_files` each investigator agent needs

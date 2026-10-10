import psycopg
from qdrant_client.http.exceptions import ApiException

from miresolver import cli
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


def test_index_reports_a_missing_database_url_cleanly(tmp_path, capsys, monkeypatch):
    models = tmp_path / "models.yaml"
    models.write_text(MODELS_YAML)
    monkeypatch.setenv("K", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # Run outside the repo so a developer's real .env cannot supply one.
    monkeypatch.chdir(tmp_path)
    exit_code = main(["index", "--models", str(models)])
    assert exit_code == 1
    assert "DATABASE_URL" in capsys.readouterr().err


class _FakeUnreachableQdrantClient:
    """Stands in for QdrantClient when the server cannot be reached.

    Any call that would hit the network raises the same exception type a
    real connection refusal surfaces as (qdrant_client wraps httpx/urllib3
    connection errors in ApiException / ResponseHandlingException, both
    subclasses of ApiException).
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def collection_exists(self, *args, **kwargs):
        raise ApiException("[Errno 61] Connection refused")

    def close(self) -> None:
        pass


def test_search_reports_an_unreachable_qdrant_cleanly(tmp_path, capsys, monkeypatch):
    models = tmp_path / "models.yaml"
    models.write_text(MODELS_YAML)
    monkeypatch.setenv("K", "secret")
    monkeypatch.setattr(cli, "QdrantClient", _FakeUnreachableQdrantClient)

    exit_code = main(
        ["search", "anything", "--models", str(models), "--qdrant-url", "http://localhost:1"]
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "search failed" in err
    assert "http://localhost:1" in err
    assert "docker compose" in err
    assert "Traceback" not in err


def test_index_reports_an_unreachable_postgres_cleanly(tmp_path, capsys, monkeypatch):
    models = tmp_path / "models.yaml"
    models.write_text(MODELS_YAML)
    monkeypatch.setenv("K", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:hunter2@localhost:1/db")

    def _fake_connect(*args, **kwargs):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(cli.psycopg, "connect", _fake_connect)

    exit_code = main(["index", "--models", str(models)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "index failed" in err
    assert "Postgres" in err
    assert "docker compose" in err
    # The password from DATABASE_URL must never reach stderr.
    assert "hunter2" not in err
    assert "Traceback" not in err


def test_redact_url_masks_a_password():
    assert cli._redact_url("postgresql://user:hunter2@localhost:5432/db") == (
        "postgresql://user:***@localhost:5432/db"
    )


def test_redact_url_does_not_invent_a_password_when_there_is_none():
    assert cli._redact_url("postgresql://user@localhost:5432/db") == (
        "postgresql://user@localhost:5432/db"
    )


def test_redact_url_never_raises_on_a_malformed_url():
    # urlsplit itself raises ValueError on some malformed IPv6-looking
    # hosts; _redact_url is called from inside a handler whose whole job
    # is to avoid a traceback, so it must swallow this and return
    # something printable rather than propagate.
    malformed = "http://[::1/badipv6"
    assert cli._redact_url(malformed) == malformed


def test_search_reads_the_api_key_from_dotenv(tmp_path, capsys, monkeypatch):
    # The README says miresolver reads .env from the working directory;
    # search must honour that for the embed tier's key, as resolve does.
    models = tmp_path / "models.yaml"
    models.write_text(MODELS_YAML)
    (tmp_path / ".env").write_text("K=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("K", raising=False)
    monkeypatch.setattr(cli, "QdrantClient", _FakeUnreachableQdrantClient)

    exit_code = main(["search", "anything", "--models", str(models)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "not set" not in err
    assert "search failed" in err


def test_index_reads_the_api_key_from_dotenv(tmp_path, capsys, monkeypatch):
    models = tmp_path / "models.yaml"
    models.write_text(MODELS_YAML)
    (tmp_path / ".env").write_text(
        "K=from-dotenv\nGITHUB_TOKEN=x\nDATABASE_URL=postgresql://u@localhost:1/db\n"
    )
    monkeypatch.chdir(tmp_path)
    for name in ("K", "GITHUB_TOKEN", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)

    def _fake_connect(*args, **kwargs):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(cli.psycopg, "connect", _fake_connect)

    exit_code = main(["index", "--models", str(models)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "not set" not in err
    assert "Postgres" in err


class _LoopRecordingEmbedder:
    """Records which event loop it was used and closed on.

    A real httpx client keeps pooled connections bound to the loop that
    opened them; closing it on a different, fresh loop raises
    "Event loop is closed" after a successful run.
    """

    def __init__(self) -> None:
        self.used_on = None
        self.closed_on = None

    async def embed(self, texts):
        import asyncio
        self.used_on = asyncio.get_running_loop()
        return [[0.0] for _ in texts]

    async def aclose(self) -> None:
        import asyncio
        self.closed_on = asyncio.get_running_loop()


class _FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeQdrantClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


def test_index_closes_the_embedder_on_the_loop_that_used_it(tmp_path, monkeypatch):
    models = tmp_path / "models.yaml"
    models.write_text(MODELS_YAML)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("K", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u@localhost:1/db")
    embedder = _LoopRecordingEmbedder()
    monkeypatch.setattr(cli, "build_embedder", lambda settings, env: embedder)
    monkeypatch.setattr(cli.psycopg, "connect", lambda *a, **k: _FakeConn())
    monkeypatch.setattr(cli, "load_documents", lambda conn: ["doc"])
    monkeypatch.setattr(cli, "QdrantClient", _FakeQdrantClient)

    async def _fake_build_index(documents, emb, index, batch_size):
        await emb.embed(["x"])
        return len(documents)

    monkeypatch.setattr(cli, "build_index", _fake_build_index)

    assert main(["index", "--models", str(models)]) == 0
    assert embedder.used_on is not None
    assert embedder.closed_on is embedder.used_on


def test_search_closes_the_embedder_on_the_loop_that_used_it(tmp_path, monkeypatch):
    models = tmp_path / "models.yaml"
    models.write_text(MODELS_YAML)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("K", "secret")
    embedder = _LoopRecordingEmbedder()
    monkeypatch.setattr(cli, "build_embedder", lambda settings, env: embedder)
    monkeypatch.setattr(cli, "QdrantClient", _FakeQdrantClient)

    async def _fake_search(query, emb, index, limit):
        await emb.embed([query])
        return []

    monkeypatch.setattr(cli, "search", _fake_search)

    assert main(["search", "anything", "--models", str(models)]) == 0
    assert embedder.used_on is not None
    assert embedder.closed_on is embedder.used_on

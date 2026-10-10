import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import psycopg
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException

from miresolver.config import Config, ConfigError, load_config, read_dotenv
from miresolver.github import GitHubClient
from miresolver.manifest import (
    Manifest,
    ManifestSizeError,
    build_manifest,
    manifest_to_json,
)
from miresolver.models.embedding import EmbeddingError
from miresolver.models.factory import build_embedder
from miresolver.models.settings import ModelConfigError, load_model_settings
from miresolver.retrieval.index import IssueIndex, default_collection_name
from miresolver.retrieval.loader import load_documents
from miresolver.retrieval.pipeline import build_index, search
from miresolver.rules import CandidateIssue, CandidatePR
from miresolver.store import create_schema, save_issues

DEFAULT_MODELS_PATH = Path("config/models.yaml")
DEFAULT_QDRANT_URL = "http://localhost:6333"


class IssueClient(Protocol):
    """Structural type for whatever fetches issues/PRs: a real GitHubClient
    or a test double. Only the methods run_ingest actually calls."""

    async def fetch_issues(
        self, limit: int
    ) -> tuple[list[CandidateIssue], dict[int, CandidatePR]]: ...

    async def aclose(self) -> None: ...


async def run_ingest(
    config: Config,
    *,
    limit: int,
    out_path: Path,
    dev_size: int,
    full_size: int,
    seed: int,
    allow_undersized: bool = False,
    client: IssueClient | None = None,
    conn: psycopg.Connection | None = None,
) -> Manifest:
    owns_client = client is None
    if client is None:
        client = GitHubClient(config.github_token, config.target_repo)

    try:
        issues, prs = await client.fetch_issues(limit)
    finally:
        if owns_client:
            await client.aclose()

    if conn is not None:
        create_schema(conn)
        save_issues(conn, issues, prs)

    manifest = build_manifest(
        issues,
        prs,
        repo=config.target_repo,
        limit=limit,
        dev_size=dev_size,
        full_size=full_size,
        seed=seed,
        allow_undersized=allow_undersized,
    )
    out_path.write_text(manifest_to_json(manifest))
    return manifest


def _load_models(path: Path):
    try:
        return load_model_settings(path.read_text())
    except OSError as exc:
        raise ModelConfigError(f"cannot read model config {path}: {exc}") from exc


def _resolve_collection(args: argparse.Namespace, settings) -> str:
    if args.collection is not None:
        return args.collection
    return default_collection_name(settings.tier("embed").model)


def _redact_url(url: str) -> str:
    """Strip a password out of a connection URL before it goes to stderr.

    Error messages naming an unreachable URL are useful; a leaked
    credential in a terminal or a captured log is not worth that. This is
    called from inside the exception handlers that exist specifically to
    avoid printing a traceback, so it must be total: whatever it is given,
    including a malformed URL, it returns something printable and never
    raises. A URL that fails to parse falls back to the raw, unredacted
    string rather than blowing up the handler that called it; a password
    is only masked when one was actually present, so `user@host` is left
    as `user@host` rather than gaining a `***` that never existed.
    """
    try:
        parts = urlsplit(url)
        netloc = parts.netloc
        if "@" in netloc:
            creds, host = netloc.rsplit("@", 1)
            if ":" in creds:
                user, _, _password = creds.partition(":")
                netloc = f"{user}:***@{host}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except ValueError:
        return url


def _run_index(args: argparse.Namespace) -> int:
    settings = _load_models(args.models)
    env = {**read_dotenv(), **os.environ}
    config = load_config(env)
    collection = _resolve_collection(args, settings)

    # Built only after every step that can fail on its own (bad model
    # config, missing DATABASE_URL) has already succeeded, so a failure
    # never leaves an embedder client open with nothing to close it.
    embedder = build_embedder(settings, os.environ)
    client: QdrantClient | None = None
    try:
        # Postgres is attempted before QdrantClient is constructed: aside
        # from keeping this test-doubleable without touching the network,
        # QdrantClient's constructor kicks off a background
        # compatibility-probe connection to the Qdrant server. There is no
        # reason to pay for that probe when the corpus can't even be read
        # yet because the database is unreachable.
        try:
            with psycopg.connect(config.database_url, autocommit=True) as conn:
                documents = load_documents(conn)
        except psycopg.OperationalError as exc:
            print(
                f"index failed: cannot connect to Postgres at "
                f"{_redact_url(config.database_url)}: {exc}. Is it running? "
                f"Try `docker compose up -d postgres`.",
                file=sys.stderr,
            )
            return 1

        client = QdrantClient(url=args.qdrant_url)
        index = IssueIndex(client, collection=collection)
        try:
            indexed = asyncio.run(
                build_index(documents, embedder, index, batch_size=args.batch_size)
            )
        except ApiException as exc:
            print(
                f"index failed: cannot reach Qdrant at "
                f"{_redact_url(args.qdrant_url)}: {exc}. Is it running? "
                f"Try `docker compose up -d qdrant`.",
                file=sys.stderr,
            )
            return 1
    finally:
        asyncio.run(embedder.aclose())
        if client is not None:
            client.close()
    print(f"indexed {indexed} issues into {collection}")
    return 0


def _run_search(args: argparse.Namespace) -> int:
    settings = _load_models(args.models)
    collection = _resolve_collection(args, settings)

    embedder = build_embedder(settings, os.environ)
    client: QdrantClient | None = None
    try:
        client = QdrantClient(url=args.qdrant_url)
        index = IssueIndex(client, collection=collection)
        try:
            hits = asyncio.run(search(args.query, embedder, index, limit=args.limit))
        except ApiException as exc:
            print(
                f"search failed: cannot reach Qdrant at "
                f"{_redact_url(args.qdrant_url)}: {exc}. Is it running? "
                f"Try `docker compose up -d qdrant`.",
                file=sys.stderr,
            )
            return 1
    finally:
        asyncio.run(embedder.aclose())
        if client is not None:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="miresolver")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="fetch issues and build the manifest")
    ingest.add_argument("--limit", type=int, default=2000)
    ingest.add_argument("--out", type=Path, default=Path("eval/manifest.json"))
    ingest.add_argument("--dev-size", type=int, default=100)
    ingest.add_argument("--full-size", type=int, default=500)
    ingest.add_argument("--seed", type=int, default=20260921)
    ingest.add_argument(
        "--allow-undersized",
        action="store_true",
        help="write the manifest even when fewer issues are eligible than "
        "--full-size asks for, marking it undersized in the metadata",
    )
    ingest.add_argument(
        "--skip-db",
        action="store_true",
        help="do not connect to Postgres or persist issues; write only the "
        "manifest file",
    )

    index_cmd = sub.add_parser("index", help="embed the corpus into Qdrant")
    index_cmd.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    index_cmd.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    index_cmd.add_argument(
        "--collection",
        default=None,
        help="defaults to a name derived from the configured embed model, "
        "so switching models never mixes vectors in one collection",
    )
    index_cmd.add_argument("--batch-size", type=int, default=64)

    search_cmd = sub.add_parser("search", help="find similar past issues")
    search_cmd.add_argument("query")
    search_cmd.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    search_cmd.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    search_cmd.add_argument(
        "--collection",
        default=None,
        help="defaults to a name derived from the configured embed model, "
        "so switching models never mixes vectors in one collection",
    )
    search_cmd.add_argument("--limit", type=int, default=10)

    args = parser.parse_args(argv)

    if args.command in ("index", "search"):
        try:
            return _run_index(args) if args.command == "index" else _run_search(args)
        except (ModelConfigError, EmbeddingError, ConfigError) as exc:
            print(f"{args.command} failed: {exc}", file=sys.stderr)
            return 1

    # The README says to copy .env.example to .env and run, so read it.
    # The real environment wins over the file.
    env = {**read_dotenv(), **os.environ}
    try:
        config = load_config(env, require_database_url=not args.skip_db)
    except ConfigError as exc:
        print(
            f"configuration error: {exc}. Set it in the environment or in "
            f"a .env file (see .env.example).",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        limit=args.limit,
        out_path=args.out,
        dev_size=args.dev_size,
        full_size=args.full_size,
        seed=args.seed,
        allow_undersized=args.allow_undersized,
    )

    try:
        if args.skip_db:
            manifest = asyncio.run(run_ingest(config, conn=None, **kwargs))
        else:
            with psycopg.connect(config.database_url, autocommit=True) as conn:
                manifest = asyncio.run(run_ingest(config, conn=conn, **kwargs))
    except ManifestSizeError as exc:
        print(
            f"ingest failed: requested dev_size={args.dev_size}, "
            f"full_size={args.full_size} (limit={args.limit}) but {exc}",
            file=sys.stderr,
        )
        return 1

    meta = manifest.meta
    if meta.undersized:
        print(
            f"UNDERSIZED: only {meta.total_eligible} eligible cases, "
            f"fewer than the {meta.full_size} requested. The manifest is "
            f"labelled undersized; do not report it as a "
            f"{meta.full_size}-case run.",
            file=sys.stderr,
        )

    print(f"examined issues: {meta.total_examined}")
    print(f"eligible cases: {meta.total_eligible}")
    print(f"selected: full={len(manifest.full)}, dev={len(manifest.dev)}")
    for reason, count in sorted(manifest.rejected.items()):
        print(f"  rejected {reason}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

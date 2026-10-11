import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import psycopg
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException

from miresolver import runtime
from miresolver.agents.graph import build_graph, resolve, retrieve_candidates
from miresolver.agents.types import NewIssue, Resolution, ResolverOptions
from miresolver.code.reader import CodeReadError

from miresolver.benchmark import (
    BenchmarkError,
    benchmark_to_json,
    build_benchmark,
    load_corpus,
)
from miresolver.config import Config, ConfigError, load_config, read_dotenv
from miresolver.evaluation.calibration import (
    CalibrationError,
    choose_thresholds,
    evaluate_thresholds,
    load_run,
)
from miresolver.evaluation.corpus import load_issue_text
from miresolver.evaluation.runner import (
    EvaluationError,
    estimate_cost,
    run_evaluation,
    run_retrieval_check,
    select_cases,
)
from miresolver.github import GitHubClient
from miresolver.manifest import (
    Manifest,
    ManifestSizeError,
    build_manifest,
    manifest_to_json,
)
from miresolver.models.chat import ChatError
from miresolver.models.embedding import EmbeddingError
from miresolver.models.factory import build_embedder
from miresolver.models.settings import ModelConfigError, load_model_settings
from miresolver.retrieval.index import (
    IssueIndex,
    KeywordIndexMissing,
    default_collection_name,
)
from miresolver.retrieval.loader import load_documents
from miresolver.retrieval.pipeline import build_index, search
from miresolver.rules import CandidateIssue, CandidatePR
from miresolver.store import create_schema, save_issues

DEFAULT_MODELS_PATH = Path("config/models.yaml")
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_MANIFEST_PATH = Path("eval/manifest.json")
DEFAULT_RESULTS_ROOT = Path("eval/results")
DEFAULT_CODE_CACHE = Path(".cache/code")
DEFAULT_MAX_COST = 3.0


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


async def _then_close(embedder, work):
    """Await work, then close the embedder on the same event loop.

    The embedder's HTTP client pools connections bound to the loop that
    opened them, so closing it from a second asyncio.run fails with
    "Event loop is closed" even after the work succeeded.
    """
    try:
        return await work
    finally:
        await embedder.aclose()


def _run_index(args: argparse.Namespace) -> int:
    settings = _load_models(args.models)
    env = _env()
    config = load_config(env)
    collection = _resolve_collection(args, settings)

    # Built only after every step that can fail on its own (bad model
    # config, missing DATABASE_URL) has already succeeded, so a failure
    # never leaves an embedder client open with nothing to close it.
    embedder = build_embedder(settings, env)
    embedder_closed = False
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
            # _then_close closes the embedder inside the loop, success or not.
            embedder_closed = True
            indexed = asyncio.run(
                _then_close(
                    embedder,
                    build_index(documents, embedder, index, batch_size=args.batch_size),
                )
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
        if not embedder_closed:
            asyncio.run(embedder.aclose())
        if client is not None:
            client.close()
    print(f"indexed {indexed} issues into {collection}")
    return 0


def _run_search(args: argparse.Namespace) -> int:
    settings = _load_models(args.models)
    collection = _resolve_collection(args, settings)

    embedder = build_embedder(settings, _env())
    embedder_closed = False
    client: QdrantClient | None = None
    try:
        client = QdrantClient(url=args.qdrant_url)
        index = IssueIndex(client, collection=collection)
        try:
            embedder_closed = True  # _then_close closes it inside the loop
            hits = asyncio.run(
                _then_close(embedder, search(args.query, embedder, index, limit=args.limit))
            )
        except ApiException as exc:
            print(
                f"search failed: cannot reach Qdrant at "
                f"{_redact_url(args.qdrant_url)}: {exc}. Is it running? "
                f"Try `docker compose up -d qdrant`.",
                file=sys.stderr,
            )
            return 1
    finally:
        if not embedder_closed:
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


def _env() -> dict[str, str]:
    return {**read_dotenv(), **os.environ}


def _load_case_texts(
    database_url: str, numbers: list[int]
) -> dict[int, tuple[str, str]]:
    with psycopg.connect(database_url, autocommit=True) as conn:
        return load_issue_text(conn, numbers)


def _database_url(env: dict[str, str], command: str) -> str:
    url = env.get("DATABASE_URL")
    if not url:
        raise ConfigError(
            f"DATABASE_URL is required: {command} reads issue text from the "
            f"Postgres corpus"
        )
    return url


def _load_corpus(database_url: str):
    with psycopg.connect(database_url, autocommit=True) as conn:
        return load_corpus(conn)


def _issue_numbers_in_run(run_dir: Path) -> set[int]:
    path = run_dir / "cases.jsonl"
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise BenchmarkError(f"cannot read {path}: {exc}") from exc
    return {int(json.loads(line)["issue_number"]) for line in lines if line.strip()}


def _run_benchmark(args: argparse.Namespace) -> int:
    if args.out.exists():
        print(
            f"benchmark failed: {args.out} exists. A published benchmark is "
            f"frozen; write a new version to a new file instead.",
            file=sys.stderr,
        )
        return 1
    env = _env()
    url = _database_url(env, "benchmark")
    excluded: set[int] = set()
    for run_dir in args.exclude_run:
        excluded |= _issue_numbers_in_run(run_dir)
    issues, prs = _load_corpus(url)
    bench = build_benchmark(
        issues, prs,
        repo=env.get("TARGET_REPO", "microsoft/vscode"),
        seed=args.seed,
        test_size=args.test_size,
        dev_size=args.dev_size,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        exclude_from_test=excluded,
        source="postgres corpus: every stored issue, v1 eligibility rules",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(benchmark_to_json(bench))
    meta = bench.meta
    print(f"examined issues: {meta['total_examined']}")
    print(f"eligible cases: {meta['total_eligible']}")
    for reason, count in sorted(bench.rejected.items()):
        print(f"  rejected {reason}: {count}")
    print(f"test: {len(bench.test)} (seed {meta['seed']}, created {meta['created_at']})")
    print(f"dev: {len(bench.dev)}")
    print(f"excluded from test as already used: {len(meta['excluded_from_test'])}")
    print(f"test_sha256: {meta['test_sha256']}")
    return 0


def _run_calibrate(args: argparse.Namespace) -> int:
    try:
        summary, signals = load_run(args.run)
    except (OSError, ValueError, KeyError) as exc:
        raise CalibrationError(f"cannot read run {args.run}: {exc}") from exc
    run = summary.get("run", {})
    if run.get("set") != "dev":
        raise CalibrationError(
            f"refusing to calibrate on a {run.get('set')!r} run; thresholds are "
            f"chosen on the dev set only"
        )
    if run.get("mode") != "always-answer":
        raise CalibrationError(
            "calibration needs an always-answer run, where every case was "
            "synthesized and its confidences recorded"
        )
    chosen = choose_thresholds(signals)
    default = evaluate_thresholds(signals, 0.5, 0.5, 0)
    print(json.dumps({"run": str(args.run), "objective":
                      "maximise answered_correct - answered_wrong; ties: more "
                      "answers, then lower thresholds",
                      "chosen": chosen, "default_0.5_0.5": default}, indent=2))
    return 0


def _options(args: argparse.Namespace) -> ResolverOptions:
    return ResolverOptions(
        retrieve_k=args.retrieve_k,
        investigate_n=args.investigators,
        abstain_threshold=args.abstain_threshold,
        synthesis_threshold=args.synthesis_threshold,
        investigator_timeout_s=args.investigator_timeout,
        always_answer=args.mode == "always-answer",
        min_agreement=args.min_agreement,
        retrieval=args.retrieval,
    )


def _runtime_kwargs(args: argparse.Namespace, settings, env) -> dict:
    return dict(
        qdrant_url=args.qdrant_url,
        collection=_resolve_collection(args, settings),
        repo=args.repo or env.get("TARGET_REPO", "microsoft/vscode"),
        local_repo=args.local_repo,
        code_cache=args.code_cache,
        options=_options(args),
    )


def _resolution_json(res: Resolution) -> dict:
    return {
        "abstained": res.abstained,
        "abstain_reason": res.abstain_reason,
        "confidence": res.confidence,
        "files": list(res.files),
        "approach": res.approach,
        "citations": [asdict(c) for c in res.citations],
        "verdicts": [
            {"issue_number": v.issue_number, "applies": v.applies,
             "confidence": v.confidence, "files": list(v.files), "error": v.error}
            for v in res.verdicts
        ],
        "tokens": {t: asdict(u) for t, u in res.usage.items()},
    }


def _print_resolution(res: Resolution) -> None:
    if res.abstained:
        print(f"No confident match ({res.abstain_reason}).")
        considered = ", ".join(f"#{v.issue_number}" for v in res.verdicts)
        if considered:
            print(f"Precedents investigated and judged insufficient: {considered}")
        return
    print(f"Confidence: {res.confidence:.2f}")
    print("Suggested files, most likely first:")
    for rank, path in enumerate(res.files, start=1):
        print(f"  {rank}. {path}")
    print(f"Approach: {res.approach}")
    print("Based on:")
    for c in res.citations:
        print(f"  #{c.issue_number}, fixed by PR {c.pr_number}")


def _run_resolve(args: argparse.Namespace) -> int:
    settings = _load_models(args.models)
    env = _env()
    if args.issue is not None:
        texts = _load_case_texts(_database_url(env, "resolve --issue"), [args.issue])
        if args.issue not in texts:
            print(f"resolve failed: issue {args.issue} is not in the corpus",
                  file=sys.stderr)
            return 1
        title, body = texts[args.issue]
    elif args.title:
        title = args.title
        body = args.body_file.read_text() if args.body_file else (args.body or "")
    else:
        print("resolve failed: give an issue title, or --issue NUMBER",
              file=sys.stderr)
        return 1
    issue = NewIssue(title, body, args.base_commit, args.issue)

    async def go() -> Resolution:
        async with runtime.open_runtime(
            settings, env, **_runtime_kwargs(args, settings, env)
        ) as deps:
            return await resolve(issue, deps)

    res = asyncio.run(go())
    if args.json:
        print(json.dumps(_resolution_json(res), indent=2))
    else:
        _print_resolution(res)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    settings = _load_models(args.models)
    try:
        manifest = json.loads(args.manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read manifest {args.manifest}: {exc}") from exc
    cases = select_cases(
        manifest, which=args.set, limit=args.limit, seed=args.seed,
        allow_full=args.allow_full, allow_test=args.final,
    )
    options = _options(args)
    if args.estimate:
        print(json.dumps(estimate_cost(len(cases), settings, options), indent=2))
        return 0

    env = _env()
    texts = _load_case_texts(
        _database_url(env, "evaluate"), [c.issue_number for c in cases]
    )
    meta = manifest.get("meta", {})
    run_meta = {
        "set": args.set,
        "seed": args.seed,
        "limit": args.limit,
        "mode": args.mode,
        "tag": args.tag or (args.mode if args.mode != "abstain" else None),
        "manifest": str(args.manifest),
        "manifest_version": meta.get("version", 1),
        "manifest_generated_at": meta.get("generated_at"),
        "manifest_created_at": meta.get("created_at"),
        "manifest_seed": meta.get("seed"),
        "test_sha256": meta.get("test_sha256") if args.set == "test" else None,
        "options": asdict(options),
        "estimate": estimate_cost(len(cases), settings, options),
    }

    if args.retrieval_only:
        async def check() -> Path:
            async with runtime.open_runtime(
                settings, env, **_runtime_kwargs(args, settings, env)
            ) as deps:
                async def retrieve(issue, exclude, k):
                    return await retrieve_candidates(
                        deps.retriever, issue, k=k, exclude=exclude
                    )

                return await run_retrieval_check(
                    cases, issue_text=texts.get, retrieve=retrieve,
                    out_root=args.out_root,
                    run_meta={k: v for k, v in run_meta.items() if k != "estimate"},
                    progress=lambda line: print(line, file=sys.stderr, flush=True),
                )

        out_dir = asyncio.run(check())
        summary = json.loads((out_dir / "summary.json").read_text())
        print(f"results written to {out_dir}")
        print(f"precedent available at depth: {summary['precedent_available_at']}")
        return 0

    async def go() -> Path:
        async with runtime.open_runtime(
            settings, env, **_runtime_kwargs(args, settings, env)
        ) as deps:
            graph = build_graph(deps)

            async def resolver(issue, exclude):
                return await resolve(issue, deps, exclude=exclude, graph=graph)

            return await run_evaluation(
                cases, issue_text=texts.get, resolver=resolver, settings=settings,
                out_root=args.out_root, run_meta=run_meta, max_cost=args.max_cost,
                progress=lambda line: print(line, file=sys.stderr, flush=True),
            )

    out_dir = asyncio.run(go())
    summary = json.loads((out_dir / "summary.json").read_text())
    metrics = summary["metrics"]
    print(f"results written to {out_dir}")
    print(f"scored {summary['counts']['scored']} of {summary['counts']['selected']} cases")
    print(f"hit@3 (all cases): {metrics['all_cases']['hit_at_3']}")
    print(f"abstention rate: {metrics['abstention_rate']}")
    print(f"accuracy when answering: {metrics['accuracy_when_answering']}")
    print(f"priced spend: ${summary['spend']['priced_total']}")
    return 0


def _add_runtime_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=None)
    parser.add_argument(
        "--repo", default=None,
        help="owner/name whose code is read; defaults to TARGET_REPO or microsoft/vscode",
    )
    parser.add_argument(
        "--local-repo", type=Path, default=None,
        help="read code from this local clone with git instead of fetching it",
    )
    parser.add_argument("--code-cache", type=Path, default=DEFAULT_CODE_CACHE)
    parser.add_argument("--retrieve-k", type=int, default=20)
    parser.add_argument("--investigators", type=int, default=3)
    parser.add_argument("--abstain-threshold", type=float, default=0.85)
    parser.add_argument(
        "--synthesis-threshold", type=float, default=0.9,
        help="minimum synthesizer confidence to answer",
    )
    parser.add_argument(
        "--min-agreement", type=int, default=2,
        help="investigators that must name one of the top three files to answer",
    )
    parser.add_argument(
        "--mode", choices=("abstain", "always-answer"), default="abstain",
        help="always-answer never abstains, for coverage-style metrics",
    )
    parser.add_argument(
        "--retrieval", choices=("hybrid", "dense", "keyword"), default="hybrid",
    )
    parser.add_argument("--investigator-timeout", type=float, default=90.0)


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

    bench_cmd = sub.add_parser(
        "benchmark",
        help="build a versioned dev/test manifest from the stored corpus",
    )
    bench_cmd.add_argument("--out", type=Path, default=Path("eval/manifest.v2.json"))
    bench_cmd.add_argument("--seed", type=int, default=20261010)
    bench_cmd.add_argument("--test-size", type=int, default=100)
    bench_cmd.add_argument("--dev-size", type=int, default=100)
    bench_cmd.add_argument(
        "--exclude-run", type=Path, action="append", default=[],
        help="results directory of an earlier run; its cases never enter the test set",
    )

    cal_cmd = sub.add_parser(
        "calibrate",
        help="choose abstention thresholds from a dev always-answer run",
    )
    cal_cmd.add_argument("run", type=Path, help="results directory of the run")

    resolve_cmd = sub.add_parser(
        "resolve", help="suggest files and an approach for one bug report"
    )
    resolve_cmd.add_argument("title", nargs="?", default=None)
    resolve_cmd.add_argument("--body", default=None)
    resolve_cmd.add_argument("--body-file", type=Path, default=None)
    resolve_cmd.add_argument(
        "--issue", type=int, default=None,
        help="read title and body of this issue from the Postgres corpus",
    )
    resolve_cmd.add_argument(
        "--base-commit", required=True,
        help="commit whose code the investigators read",
    )
    resolve_cmd.add_argument("--json", action="store_true")
    _add_runtime_flags(resolve_cmd)

    eval_cmd = sub.add_parser(
        "evaluate", help="score the resolver against the manifest answer key"
    )
    eval_cmd.add_argument("--set", choices=("dev", "test", "full"), default="dev")
    eval_cmd.add_argument("--limit", type=int, default=None)
    eval_cmd.add_argument("--seed", type=int, default=20260921)
    eval_cmd.add_argument("--allow-full", action="store_true")
    eval_cmd.add_argument(
        "--final", action="store_true",
        help="required for --set test: the one held-out run per mode",
    )
    eval_cmd.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST)
    eval_cmd.add_argument("--estimate", action="store_true")
    eval_cmd.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    eval_cmd.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    eval_cmd.add_argument(
        "--retrieval-only", action="store_true",
        help="score retrieval alone (precedent availability by depth); no chat calls",
    )
    eval_cmd.add_argument("--tag", default=None, help="suffix for the run directory")
    _add_runtime_flags(eval_cmd)

    args = parser.parse_args(argv)

    if args.command in ("resolve", "evaluate"):
        try:
            if args.command == "resolve":
                return _run_resolve(args)
            return _run_evaluate(args)
        except (
            ModelConfigError, EmbeddingError, ConfigError, ChatError,
            EvaluationError, CodeReadError, KeywordIndexMissing,
        ) as exc:
            print(f"{args.command} failed: {exc}", file=sys.stderr)
            return 1
        except (ApiException, ResponseHandlingException) as exc:
            print(
                f"{args.command} failed: cannot reach Qdrant at "
                f"{_redact_url(args.qdrant_url)}: {exc}. Is it running? "
                f"Try `docker compose up -d qdrant`.",
                file=sys.stderr,
            )
            return 1
        except psycopg.OperationalError as exc:
            print(
                f"{args.command} failed: cannot connect to Postgres: {exc}. "
                f"Is it running? Try `docker compose up -d postgres`.",
                file=sys.stderr,
            )
            return 1

    if args.command == "calibrate":
        try:
            return _run_calibrate(args)
        except CalibrationError as exc:
            print(f"calibrate failed: {exc}", file=sys.stderr)
            return 1

    if args.command == "benchmark":
        try:
            return _run_benchmark(args)
        except (ConfigError, BenchmarkError) as exc:
            print(f"benchmark failed: {exc}", file=sys.stderr)
            return 1
        except psycopg.OperationalError as exc:
            print(
                f"benchmark failed: cannot connect to Postgres: {exc}. "
                f"Is it running? Try `docker compose up -d postgres`.",
                file=sys.stderr,
            )
            return 1

    if args.command in ("index", "search"):
        try:
            return _run_index(args) if args.command == "index" else _run_search(args)
        except (
            ModelConfigError, EmbeddingError, ConfigError, KeywordIndexMissing,
        ) as exc:
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

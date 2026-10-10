"""Run the resolver over manifest cases and write scored results.

Output for one run lands in eval/results/{set}-n{count}-seed{seed}-{UTC}/:
summary.json (metrics, models, counts, tokens, spend) and cases.jsonl (one
line per case, written as each case finishes so an interrupted run keeps
what it already paid for).
"""

import json
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from miresolver.agents.prompts import (
    INVESTIGATE_SYSTEM,
    ISSUE_BODY_CHARS,
    SCREEN_SYSTEM,
    SYNTHESIZE_SYSTEM,
)
from miresolver.agents.investigate import INVESTIGATE_MAX_TOKENS
from miresolver.agents.screen import SCREEN_MAX_TOKENS
from miresolver.agents.synthesize import SYNTHESIZE_MAX_TOKENS
from miresolver.agents.types import Candidate, NewIssue, Resolution, ResolverOptions
from miresolver.benchmark import digest_cases
from miresolver.evaluation.metrics import score_case, summarize
from miresolver.manifest import EvalCase
from miresolver.models.chat import Usage
from miresolver.models.factory import CHAT_TIERS
from miresolver.models.settings import ModelSettings
from miresolver.retrieval.documents import MAX_EMBED_CHARS

CHARS_PER_TOKEN = 4
#: Rough allowance for prompt scaffolding beyond the measured text budgets.
PROMPT_OVERHEAD_CHARS = 800

Exclude = Callable[[Candidate], bool]
Resolver = Callable[[NewIssue, Exclude], Awaitable[Resolution]]
IssueText = Callable[[int], tuple[str, str] | None]


class EvaluationError(Exception):
    """Raised for an evaluation request that should not run."""


def select_cases(
    manifest: Mapping,
    *,
    which: str,
    limit: int | None,
    seed: int,
    allow_full: bool = False,
    allow_test: bool = False,
) -> list[EvalCase]:
    """Pick the cases to run, reproducibly.

    A limit draws a seeded sample, sorted by issue number, so the same seed
    always names the same subset. The full set needs allow_full, so a large
    paid run cannot start by accident. The held-out test set needs
    allow_test, always runs whole, and is checked against the digest
    recorded when it was frozen.
    """
    if which not in ("dev", "full", "test"):
        raise EvaluationError(f"unknown case set {which!r}; use dev, test or full")
    if limit is not None and limit < 1:
        raise EvaluationError(f"--limit must be at least 1, got {limit}")
    if which == "full" and not allow_full:
        raise EvaluationError(
            "refusing to run the full set without --allow-full; it is the "
            "published run and the most expensive one"
        )
    if which == "test":
        if not allow_test:
            raise EvaluationError(
                "refusing to run the held-out test set without --final; it is "
                "run once per mode with the final configuration, never for tuning"
            )
        if limit is not None:
            raise EvaluationError(
                "the held-out test set always runs whole; drop --limit"
            )
    if which not in manifest:
        raise EvaluationError(f"this manifest has no {which!r} set")
    if which == "test":
        recorded = manifest.get("meta", {}).get("test_sha256")
        if recorded is not None and digest_cases(manifest["test"]) != recorded:
            raise EvaluationError(
                "the test set no longer matches the digest recorded when it "
                "was frozen; refusing to score an edited test set"
            )
    cases = [
        EvalCase(
            issue_number=int(c["issue_number"]),
            pr_number=int(c["pr_number"]),
            base_commit=str(c["base_commit"]),
            changed_files=tuple(c["changed_files"]),
        )
        for c in manifest[which]
    ]
    if limit is not None and limit < len(cases):
        cases = sorted(
            random.Random(seed).sample(cases, limit), key=lambda c: c.issue_number
        )
    return cases


def leakage_filter(case: EvalCase) -> Exclude:
    """Exclude candidates that could carry the case's own answer.

    Drops the case's issue, anything fixed by the case's PR, and anything
    fixed by a PR numbered at or above it. PR numbers are assigned when a
    PR is opened, so a PR opened after the fix cannot have been its
    precedent. Candidates without a known fix are dropped too: they carry
    no resolution to learn from.
    """

    def exclude(candidate: Candidate) -> bool:
        return (
            candidate.issue_number == case.issue_number
            or candidate.pr_number is None
            or candidate.pr_number >= case.pr_number
        )

    return exclude


def _tokens(chars: int) -> int:
    return -(-chars // CHARS_PER_TOKEN)


def estimate_cost(
    n_cases: int, settings: ModelSettings, options: ResolverOptions
) -> dict:
    """Upper-bound spend for n cases, from the prompt budgets alone.

    Assumes every prompt is filled to its character budget and every reply
    uses its full token cap, at roughly four characters per token. Real
    runs should come in under this; the summary records actual spend.
    """
    issue = ISSUE_BODY_CHARS + 200
    per_case = {
        "embed": (_tokens(MAX_EMBED_CHARS), 0),
        "screen": (
            _tokens(
                len(SCREEN_SYSTEM) + PROMPT_OVERHEAD_CHARS + issue
                + options.retrieve_k * (options.candidate_body_chars + 200)
            ),
            SCREEN_MAX_TOKENS,
        ),
        "investigate": (
            options.investigate_n
            * _tokens(
                len(INVESTIGATE_SYSTEM) + PROMPT_OVERHEAD_CHARS + 2 * issue
                + options.max_files_per_candidate * (options.file_char_budget + 200)
            ),
            options.investigate_n * INVESTIGATE_MAX_TOKENS,
        ),
        "synthesize": (
            _tokens(
                len(SYNTHESIZE_SYSTEM) + PROMPT_OVERHEAD_CHARS + issue
                + options.investigate_n * 2500
            ),
            SYNTHESIZE_MAX_TOKENS,
        ),
    }
    tiers: dict[str, dict] = {}
    for name, (inp, out) in per_case.items():
        usage = Usage(inp * n_cases, out * n_cases)
        tier = settings.tier(name)
        tiers[name] = {
            "model": tier.model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "dollars": _round(tier.cost(usage)),
        }
    priced = [t["dollars"] for t in tiers.values() if t["dollars"] is not None]
    return {
        "cases": n_cases,
        "assumption": "upper bound: full prompt budgets, full reply caps, "
        f"{CHARS_PER_TOKEN} characters per token",
        "tiers": tiers,
        "priced_dollars": _round(sum(priced)),
        "unpriced_tiers": [n for n, t in tiers.items() if t["dollars"] is None],
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _case_record(case: EvalCase, status: str, **extra: object) -> dict:
    return {
        "issue_number": case.issue_number,
        "pr_number": case.pr_number,
        "base_commit": case.base_commit,
        "answer_files": list(case.changed_files),
        "status": status,
        **extra,
    }


def _resolution_fields(res: Resolution) -> dict:
    return {
        "abstained": res.abstained,
        "abstain_reason": res.abstain_reason,
        "confidence": res.confidence,
        "predicted_files": list(res.files),
        "approach": res.approach,
        "citations": [asdict(c) for c in res.citations],
        "candidates": [
            {"issue_number": c.issue_number, "pr_number": c.pr_number}
            for c in res.candidates
        ],
        "screened": [
            {"issue_number": c.issue_number, "screen_score": c.screen_score}
            for c in res.screened
        ],
        "verdicts": [
            {
                "issue_number": v.issue_number,
                "applies": v.applies,
                "confidence": v.confidence,
                "files": list(v.files),
                "error": v.error,
            }
            for v in res.verdicts
        ],
        "synthesis_fallback": res.synthesis_fallback,
        "unmetered_calls": res.unmetered_calls,
        "tokens": {t: asdict(u) for t, u in res.usage.items()},
    }


def _run_dir(
    out_root: Path, run_meta: Mapping[str, object], n: int,
    timestamp: str | None, prefix: str = "",
) -> Path:
    stamp = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = run_meta.get("tag")
    out_dir = Path(out_root) / (
        f"{prefix}{run_meta.get('set', 'cases')}-n{n}-"
        f"seed{run_meta.get('seed')}-{stamp}" + (f"-{tag}" if tag else "")
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _first_precedent_rank(
    candidates: Sequence[Candidate], answer: Sequence[str]
) -> int | None:
    key = set(answer)
    for rank, c in enumerate(candidates, start=1):
        if key.intersection(c.changed_files):
            return rank
    return None


Retrieve = Callable[[NewIssue, Exclude, int], Awaitable[Sequence[Candidate]]]


async def run_retrieval_check(
    cases: Sequence[EvalCase],
    *,
    issue_text: IssueText,
    retrieve: Retrieve,
    out_root: Path,
    run_meta: Mapping[str, object],
    depths: Sequence[int] = (1, 3, 5, 10, 20, 40),
    timestamp: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Retrieval alone, with the same leakage filter and no chat calls.

    For each case, records the rank of the first candidate whose fix
    touched an answer-key file, and reports the share of cases with such a
    precedent within each depth. Costs only the query embeddings, so
    retrieval settings can be compared before any agent run.
    """
    out_dir = _run_dir(out_root, run_meta, len(cases), timestamp, prefix="retrieval-")
    deepest = max(depths)
    ranks: list[int | None] = []
    counts = {"selected": len(cases), "scored": 0, "skipped_no_text": 0}
    with (out_dir / "cases.jsonl").open("w") as log:
        for i, case in enumerate(cases):
            text = issue_text(case.issue_number)
            if text is None:
                counts["skipped_no_text"] += 1
                record = _case_record(case, "skipped_no_text")
            else:
                issue = NewIssue(text[0], text[1], case.base_commit, case.issue_number)
                found = await retrieve(issue, leakage_filter(case), deepest)
                rank = _first_precedent_rank(found, case.changed_files)
                ranks.append(rank)
                counts["scored"] += 1
                record = _case_record(
                    case, "scored", first_precedent_rank=rank,
                    candidates=[
                        {"issue_number": c.issue_number, "pr_number": c.pr_number}
                        for c in found
                    ],
                )
            log.write(json.dumps(record, sort_keys=True) + "\n")
            if progress:
                progress(f"[{i + 1}/{len(cases)}] #{case.issue_number} "
                         f"first precedent at {record.get('first_precedent_rank')}")
    n = len(ranks)
    summary = {
        "run": dict(run_meta),
        "counts": counts,
        "precedent_available_at": {
            str(d): (round(sum(r is not None and r <= d for r in ranks) / n, 4) if n else None)
            for d in depths
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return out_dir


async def run_evaluation(
    cases: Sequence[EvalCase],
    *,
    issue_text: IssueText,
    resolver: Resolver,
    settings: ModelSettings,
    out_root: Path,
    run_meta: Mapping[str, object],
    max_cost: float | None = None,
    timestamp: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    out_dir = _run_dir(out_root, run_meta, len(cases), timestamp)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    scores, latencies = [], []
    usage: dict[str, Usage] = {t: Usage() for t in CHAT_TIERS}
    counts = {"selected": len(cases), "scored": 0, "skipped_no_text": 0,
              "errors": 0, "not_run": 0}
    stopped_early = False
    unmetered_calls = 0

    def spent() -> float:
        return sum(settings.tier(t).cost(u) or 0.0 for t, u in usage.items())

    with (out_dir / "cases.jsonl").open("w") as log:
        for i, case in enumerate(cases):
            if max_cost is not None and spent() >= max_cost:
                stopped_early = True
                counts["not_run"] = len(cases) - i
                break
            text = issue_text(case.issue_number)
            if text is None:
                counts["skipped_no_text"] += 1
                record = _case_record(case, "skipped_no_text")
            else:
                issue = NewIssue(text[0], text[1], case.base_commit, case.issue_number)
                started = time.perf_counter()
                try:
                    res = await resolver(issue, leakage_filter(case))
                except Exception as exc:  # record and keep going
                    counts["errors"] += 1
                    record = _case_record(
                        case, "error", error=f"{type(exc).__name__}: {exc}"
                    )
                else:
                    latency = time.perf_counter() - started
                    unmetered_calls += res.unmetered_calls
                    for tier, u in res.usage.items():
                        if tier in usage:
                            usage[tier] = usage[tier] + u
                    score = score_case(
                        res.files, case.changed_files, abstained=res.abstained,
                        candidate_files=[f for c in res.candidates for f in c.changed_files],
                    )
                    scores.append(score)
                    latencies.append(latency)
                    counts["scored"] += 1
                    record = _case_record(
                        case, "scored", latency_s=round(latency, 3),
                        hit_at_1=score.hit_at_1, hit_at_3=score.hit_at_3,
                        precision_at_3=round(score.precision_at_3, 4),
                        recall_at_3=round(score.recall_at_3, 4),
                        precedent_available=score.precedent_available,
                        **_resolution_fields(res),
                    )
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            if progress:
                progress(
                    f"[{i + 1}/{len(cases)}] #{case.issue_number} {record['status']}"
                    f" hit@3={record.get('hit_at_3')} abstained={record.get('abstained')}"
                    f" spent=${spent():.4f}"
                )

    tier_costs = {t: _round(settings.tier(t).cost(u)) for t, u in usage.items()}
    summary = {
        "run": {**run_meta, "started_at": started_at,
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds")},
        "models": {
            t: {"provider": settings.tier(t).provider, "model": settings.tier(t).model}
            for t in ("embed", *CHAT_TIERS)
        },
        "counts": counts,
        "stopped_early": stopped_early,
        "metrics": summarize(scores, latencies),
        "tokens": {t: asdict(u) for t, u in usage.items()},
        "spend": {
            "tiers": tier_costs,
            "priced_total": _round(sum(c for c in tier_costs.values() if c is not None)),
            "unpriced_tiers": [t for t, c in tier_costs.items() if c is None],
            "max_cost": max_cost,
            "unmetered": {"error_cases": counts["errors"], "model_calls": unmetered_calls},
            "note": "priced_total is a floor: query embeddings, error cases and "
            "timed-out or crashed model calls are not metered; see the estimate "
            "for an upper bound",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return out_dir

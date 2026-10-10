import json
import random
import types
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from miresolver.rules import (
    MAX_CHANGED_FILES,
    MIN_CHANGED_FILES,
    CandidateIssue,
    CandidatePR,
    is_eligible,
    source_files,
)


@dataclass(frozen=True)
class EvalCase:
    issue_number: int
    pr_number: int
    base_commit: str
    changed_files: tuple[str, ...]


@dataclass(frozen=True)
class ManifestMeta:
    """Everything a reader of the committed file needs to re-derive it.

    Without this a reader cannot recover which repository was read, which
    seed produced the sample, or how large the eligible pool actually was,
    so the reproducibility claim in the README is unverifiable.
    """

    repo: str
    seed: int
    dev_size: int
    full_size: int
    limit: int
    min_changed_files: int
    max_changed_files: int
    total_examined: int
    total_eligible: int
    undersized: bool
    generated_at: str


@dataclass(frozen=True)
class Manifest:
    dev: tuple[EvalCase, ...]
    full: tuple[EvalCase, ...]
    rejected: Mapping[str, int]
    meta: ManifestMeta


class ManifestSizeError(ValueError):
    """The requested dev/full sizes cannot be satisfied.

    A ValueError subclass so existing callers keep working, but distinct so
    the CLI can tell a sizing problem from a ValueError raised anywhere else
    in the run.
    """


def eligible_cases(
    issues: Sequence[CandidateIssue], prs: Mapping[int, CandidatePR]
) -> tuple[list[EvalCase], Counter[str]]:
    """Every eligible case, in issue-number order, and rejections by reason.

    The single place eligibility and the answer key are derived, shared by
    every manifest version so they cannot drift apart.
    """
    eligible: list[EvalCase] = []
    rejected: Counter[str] = Counter()
    for issue in sorted(issues, key=lambda i: i.number):
        verdict = is_eligible(issue, prs)
        if not verdict.eligible:
            rejected[verdict.reason] += 1
            continue
        pr = next(
            prs[n] for n in issue.linked_pr_numbers if n in prs and prs[n].merged
        )
        eligible.append(
            EvalCase(
                issue_number=issue.number,
                pr_number=pr.number,
                base_commit=pr.base_commit,
                changed_files=source_files(pr.changed_files),
            )
        )
    return eligible, rejected


def build_manifest(
    issues: Sequence[CandidateIssue],
    prs: Mapping[int, CandidatePR],
    *,
    repo: str,
    limit: int,
    dev_size: int,
    full_size: int,
    seed: int,
    allow_undersized: bool = False,
    generated_at: str | None = None,
) -> Manifest:
    if dev_size >= full_size:
        raise ManifestSizeError("dev_size must be strictly less than full_size")

    eligible, rejected = eligible_cases(issues, prs)

    undersized = len(eligible) < full_size
    if undersized and not allow_undersized:
        raise ManifestSizeError(
            f"not enough eligible issues: full_size={full_size} was requested "
            f"but only {len(eligible)} eligible of {len(issues)} examined. "
            f"Raise --limit, or pass --allow-undersized to publish a smaller "
            f"set that is labelled as undersized in the manifest."
        )

    rng = random.Random(seed)
    full = sorted(
        rng.sample(eligible, min(full_size, len(eligible))),
        key=lambda c: c.issue_number,
    )

    if len(full) <= dev_size:
        raise ManifestSizeError(
            f"not enough eligible issues: need at least {dev_size + 1} to form "
            f"a strict subset of size {dev_size}, but only {len(full)} available"
        )

    dev = sorted(
        rng.sample(full, dev_size),
        key=lambda c: c.issue_number,
    )
    return Manifest(
        dev=tuple(dev),
        full=tuple(full),
        rejected=types.MappingProxyType(dict(rejected)),
        meta=ManifestMeta(
            repo=repo,
            seed=seed,
            dev_size=dev_size,
            full_size=full_size,
            limit=limit,
            min_changed_files=MIN_CHANGED_FILES,
            max_changed_files=MAX_CHANGED_FILES,
            total_examined=len(issues),
            total_eligible=len(eligible),
            undersized=undersized,
            generated_at=generated_at
            or datetime.now(UTC).isoformat(timespec="seconds"),
        ),
    )


def _case_to_dict(case: EvalCase) -> dict[str, object]:
    return {
        "issue_number": case.issue_number,
        "pr_number": case.pr_number,
        "base_commit": case.base_commit,
        "changed_files": list(case.changed_files),
    }


def manifest_to_json(manifest: Manifest) -> str:
    return json.dumps(
        {
            "dev": [_case_to_dict(c) for c in manifest.dev],
            "full": [_case_to_dict(c) for c in manifest.full],
            "rejected": dict(sorted(manifest.rejected.items())),
            "meta": asdict(manifest.meta),
        },
        indent=2,
        sort_keys=True,
    )

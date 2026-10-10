"""Benchmark v2: a dev set for tuning and a frozen, held-out test set.

v1 (eval/manifest.json) was built from an ingest window that has since
moved, so many of its cases have no issue text in the current corpus. v2
is rebuilt from the corpus actually stored in Postgres, with the v1
eligibility rules and answer-key derivation unchanged, so every case can be
scored. The test set is drawn first, with its own seed, and its contents
are fingerprinted so any later edit is detectable.
"""

import hashlib
import json
import random
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from miresolver.github import CHANGED_FILE_PAGE_SIZE, LINKED_PR_PAGE_SIZE
from miresolver.manifest import EvalCase, _case_to_dict, eligible_cases
from miresolver.rules import MAX_CHANGED_FILES, MIN_CHANGED_FILES, CandidateIssue, CandidatePR

VERSION = 2


class BenchmarkError(ValueError):
    """The benchmark cannot be built as requested."""


@dataclass(frozen=True)
class Benchmark:
    dev: tuple[EvalCase, ...]
    test: tuple[EvalCase, ...]
    rejected: Mapping[str, int]
    pool_size: int
    meta: Mapping[str, object] = field(default_factory=dict)


def corpus_from_rows(
    *,
    issue_rows: Iterable[tuple],
    pr_rows: Iterable[tuple],
    link_rows: Iterable[tuple[int, int]],
) -> tuple[list[CandidateIssue], dict[int, CandidatePR]]:
    """Rebuild ingest's inputs from the rows Postgres stores.

    Postgres keeps no record of whether GitHub truncated a list, so a full
    page is treated as truncated. That can only drop cases v1 would have
    kept, never admit one it would have rejected.
    """
    prs = {
        number: CandidatePR(
            number=number,
            merged=bool(merged),
            title=title,
            changed_files=tuple(files or ()),
            base_commit=base or "",
            files_truncated=len(files or ()) >= CHANGED_FILE_PAGE_SIZE,
        )
        for number, merged, title, files, base in pr_rows
    }
    links: dict[int, list[int]] = {}
    for issue_number, pr_number in link_rows:
        links.setdefault(issue_number, []).append(pr_number)
    issues = []
    for number, title, body, labels in issue_rows:
        linked = tuple(sorted(links.get(number, ())))
        issues.append(
            CandidateIssue(
                number=number,
                title=title,
                body=body,
                labels=tuple(labels or ()),
                linked_pr_numbers=linked,
                linkage_truncated=len(linked) >= LINKED_PR_PAGE_SIZE,
            )
        )
    return issues, prs


def load_corpus(conn) -> tuple[list[CandidateIssue], dict[int, CandidatePR]]:
    with conn.cursor() as cur:
        cur.execute("SELECT number, title, body, labels FROM issues")
        issue_rows = cur.fetchall()
        cur.execute(
            "SELECT number, merged, title, changed_files, base_commit FROM pull_requests"
        )
        pr_rows = cur.fetchall()
        cur.execute("SELECT issue_number, pr_number FROM issue_pr_links")
        link_rows = cur.fetchall()
    return corpus_from_rows(issue_rows=issue_rows, pr_rows=pr_rows, link_rows=link_rows)


def digest_cases(cases: Sequence[Mapping]) -> str:
    """SHA-256 over the canonical JSON of a case list."""
    canonical = json.dumps(list(cases), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_benchmark(
    issues: Sequence[CandidateIssue],
    prs: Mapping[int, CandidatePR],
    *,
    repo: str,
    seed: int,
    test_size: int,
    dev_size: int,
    created_at: str,
    exclude_from_test: Collection[int] = (),
    source: str = "",
) -> Benchmark:
    """Draw the held-out test set first, then the dev set from the rest.

    Cases named in exclude_from_test (those already run while the system
    was being developed) can be drawn for dev but never for test, so the
    test set holds only cases no one has looked at results for.
    """
    pool, rejected = eligible_cases(issues, prs)
    excluded = set(exclude_from_test)
    test_pool = [c for c in pool if c.issue_number not in excluded]
    if len(test_pool) < test_size or len(pool) < test_size + dev_size:
        raise BenchmarkError(
            f"not enough eligible cases: {len(pool)} eligible "
            f"({len(test_pool)} unused in development) for test_size="
            f"{test_size} and dev_size={dev_size}"
        )
    rng = random.Random(seed)
    test = sorted(rng.sample(test_pool, test_size), key=lambda c: c.issue_number)
    taken = {c.issue_number for c in test}
    rest = [c for c in pool if c.issue_number not in taken]
    dev = sorted(rng.sample(rest, dev_size), key=lambda c: c.issue_number)
    meta = {
        "version": VERSION,
        "repo": repo,
        "seed": seed,
        "test_size": test_size,
        "dev_size": dev_size,
        "created_at": created_at,
        "source": source,
        "total_examined": len(issues),
        "total_eligible": len(pool),
        "min_changed_files": MIN_CHANGED_FILES,
        "max_changed_files": MAX_CHANGED_FILES,
        "excluded_from_test": sorted(excluded & {c.issue_number for c in pool}),
        "test_sha256": digest_cases([_case_to_dict(c) for c in test]),
    }
    return Benchmark(
        dev=tuple(dev),
        test=tuple(test),
        rejected=MappingProxyType(dict(rejected)),
        pool_size=len(pool),
        meta=MappingProxyType(meta),
    )


def benchmark_to_json(bench: Benchmark) -> str:
    return json.dumps(
        {
            "dev": [_case_to_dict(c) for c in bench.dev],
            "test": [_case_to_dict(c) for c in bench.test],
            "rejected": dict(sorted(bench.rejected.items())),
            "meta": dict(bench.meta),
        },
        indent=2,
        sort_keys=True,
    ) + "\n"

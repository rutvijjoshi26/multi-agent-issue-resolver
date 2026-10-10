import json

import pytest

from miresolver.benchmark import (
    BenchmarkError,
    build_benchmark,
    benchmark_to_json,
    corpus_from_rows,
    digest_cases,
)
from miresolver.github import CHANGED_FILE_PAGE_SIZE, LINKED_PR_PAGE_SIZE
from miresolver.manifest import build_manifest
from miresolver.rules import CandidateIssue, CandidatePR

REPO = "octocat/hello-world"


def make_corpus(count: int):
    issues, prs = [], {}
    for n in range(1, count + 1):
        issues.append(
            CandidateIssue(
                number=n, title=f"issue {n}", body="body", labels=("bug",),
                linked_pr_numbers=(1000 + n,),
            )
        )
        prs[1000 + n] = CandidatePR(
            number=1000 + n, merged=True, title=f"fix {n}",
            changed_files=(f"src/f{n}.ts", f"src/test/f{n}.test.ts"),
            base_commit=f"{n:040x}",
        )
    return issues, prs


def build(count=60, **kw):
    issues, prs = make_corpus(count)
    defaults = dict(
        repo=REPO, seed=7, test_size=10, dev_size=20,
        created_at="2026-10-10T00:00:00+00:00",
    )
    defaults.update(kw)
    return build_benchmark(issues, prs, **defaults)


def test_test_and_dev_are_disjoint_and_sized():
    bench = build()
    test = {c.issue_number for c in bench.test}
    dev = {c.issue_number for c in bench.dev}
    assert len(test) == 10 and len(dev) == 20
    assert not test & dev


def test_the_split_is_reproducible_from_the_seed():
    assert build().test == build().test
    assert build().dev == build().dev
    assert build(seed=8).test != build().test


def test_answer_keys_are_derived_exactly_as_in_v1():
    issues, prs = make_corpus(60)
    v1 = build_manifest(
        issues, prs, repo=REPO, limit=60, dev_size=5, full_size=60, seed=1
    )
    v1_by_issue = {c.issue_number: c for c in v1.full}
    for case in build().test + build().dev:
        assert case == v1_by_issue[case.issue_number]


def test_rejections_use_the_v1_rules():
    issues, prs = make_corpus(60)
    issues[0] = CandidateIssue(1, "t", "b", ("feature",), (1001,))
    bench = build_benchmark(
        issues, prs, repo=REPO, seed=7, test_size=10, dev_size=20,
        created_at="x",
    )
    assert bench.rejected == {"no_bug_label": 1}
    assert bench.pool_size == 59


def test_cases_already_used_in_development_never_enter_the_test_set():
    used = set(range(1, 41))
    bench = build(exclude_from_test=used)
    assert not {c.issue_number for c in bench.test} & used
    # They remain available to the dev set.
    assert {c.issue_number for c in bench.dev} & used


def test_too_small_a_pool_is_refused():
    with pytest.raises(BenchmarkError, match="eligible"):
        build(count=25)


def test_json_records_seed_time_and_a_digest_of_the_frozen_test_set():
    bench = build()
    data = json.loads(benchmark_to_json(bench))
    meta = data["meta"]
    assert meta["version"] == 2
    assert meta["seed"] == 7
    assert meta["created_at"] == "2026-10-10T00:00:00+00:00"
    assert meta["test_sha256"] == digest_cases(data["test"])
    assert set(data) == {"dev", "test", "rejected", "meta"}


def test_digest_changes_when_any_answer_key_changes():
    data = json.loads(benchmark_to_json(build()))
    before = digest_cases(data["test"])
    data["test"][0]["changed_files"].append("src/extra.ts")
    assert digest_cases(data["test"]) != before


def test_rows_with_a_full_page_of_files_are_treated_as_truncated():
    # Postgres keeps no truncation flag, so a full page is assumed to be a
    # truncated one: stricter than the GitHub-side check, never looser.
    files = tuple(f"src/f{i}.ts" for i in range(CHANGED_FILE_PAGE_SIZE))
    issues, prs = corpus_from_rows(
        issue_rows=[(1, "t", "b", ["bug"])],
        pr_rows=[(10, True, "fix", list(files), "a" * 40)],
        link_rows=[(1, 10)],
    )
    assert prs[10].files_truncated
    assert not issues[0].linkage_truncated


def test_rows_with_a_full_page_of_links_are_treated_as_truncated():
    links = [(1, 10 + i) for i in range(LINKED_PR_PAGE_SIZE)]
    prs_rows = [(10 + i, False, "x", [], "") for i in range(LINKED_PR_PAGE_SIZE)]
    issues, prs = corpus_from_rows(
        issue_rows=[(1, "t", "b", ["bug"])], pr_rows=prs_rows, link_rows=links
    )
    assert issues[0].linkage_truncated
    assert issues[0].linked_pr_numbers == tuple(10 + i for i in range(LINKED_PR_PAGE_SIZE))

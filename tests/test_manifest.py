import json

import pytest

from miresolver.manifest import (
    ManifestSizeError,
    build_manifest,
    manifest_to_json,
)
from miresolver.rules import (
    MAX_CHANGED_FILES,
    MIN_CHANGED_FILES,
    CandidateIssue,
    CandidatePR,
    source_files,
)

REPO = "octocat/hello-world"
LIMIT = 2000


def make_pair(n: int, *, labels=("bug",), merged=True, files=None):
    files = files if files is not None else (f"src/f{n}.ts",)
    issue = CandidateIssue(
        number=n, title=f"issue {n}", body="body",
        labels=labels, linked_pr_numbers=(1000 + n,),
    )
    pr = CandidatePR(
        number=1000 + n, merged=merged, title=f"fix {n}", changed_files=files,
        base_commit=f"{n:040x}",
    )
    return issue, pr


def build(count: int, **kw):
    issues, prs = [], {}
    for n in range(1, count + 1):
        issue, pr = make_pair(n, **kw)
        issues.append(issue)
        prs[pr.number] = pr
    return issues, prs


def test_dev_set_is_strict_subset_of_full_set():
    issues, prs = build(20)
    m = build_manifest(issues, prs, repo=REPO, limit=LIMIT, dev_size=5, full_size=10, seed=42)
    assert len(m.dev) == 5
    assert len(m.full) == 10
    dev_numbers = {c.issue_number for c in m.dev}
    full_numbers = {c.issue_number for c in m.full}
    assert dev_numbers < full_numbers


def test_selection_is_deterministic_for_a_seed():
    issues, prs = build(20)
    kw = dict(
        repo=REPO, limit=LIMIT, dev_size=5, full_size=10, seed=42,
        generated_at="2026-09-21T00:00:00+00:00",
    )
    a = build_manifest(issues, prs, **kw)
    b = build_manifest(issues, prs, **kw)
    assert manifest_to_json(a) == manifest_to_json(b)


def test_ineligible_issues_are_excluded_and_counted():
    issues, prs = build(10)
    bad_issue, bad_pr = make_pair(99, labels=("feature-request",))
    issues.append(bad_issue)
    prs[bad_pr.number] = bad_pr

    m = build_manifest(issues, prs, repo=REPO, limit=LIMIT, dev_size=2, full_size=5, seed=1)
    assert 99 not in {c.issue_number for c in m.full}
    assert m.rejected["no_bug_label"] == 1


def test_manifest_records_the_changed_files_as_the_answer_key():
    issues, prs = build(5)
    m = build_manifest(issues, prs, repo=REPO, limit=LIMIT, dev_size=1, full_size=2, seed=7)
    case = m.full[0]
    assert case.changed_files == source_files(prs[case.pr_number].changed_files)


def test_manifest_json_is_sorted_and_stable():
    issues, prs = build(10)
    m = build_manifest(issues, prs, repo=REPO, limit=LIMIT, dev_size=2, full_size=4, seed=3)
    parsed = json.loads(manifest_to_json(m))
    numbers = [c["issue_number"] for c in parsed["full"]]
    assert numbers == sorted(numbers)


def test_dev_size_greater_than_full_size_raises():
    issues, prs = build(20)
    with pytest.raises(ManifestSizeError, match="dev_size must be strictly less than full_size"):
        build_manifest(issues, prs, repo=REPO, limit=LIMIT, dev_size=10, full_size=5, seed=42)


def test_dev_size_equal_to_full_size_raises():
    issues, prs = build(20)
    with pytest.raises(ManifestSizeError, match="dev_size must be strictly less than full_size"):
        build_manifest(issues, prs, repo=REPO, limit=LIMIT, dev_size=5, full_size=5, seed=42)


def test_too_few_eligible_issues_raises():
    issues, prs = build(4)
    with pytest.raises(ManifestSizeError, match="not enough eligible issues"):
        build_manifest(issues, prs, repo=REPO, limit=LIMIT, dev_size=4, full_size=5, seed=42)


def test_answer_key_excludes_non_source_files():
    issues, prs = build(5)
    noisy_issue, noisy_pr = make_pair(
        42, files=("src/fix.ts", "src/test/fix.test.ts", "CHANGELOG.md")
    )
    issues.append(noisy_issue)
    prs[noisy_pr.number] = noisy_pr

    m = build_manifest(issues, prs, repo=REPO, limit=LIMIT, dev_size=1, full_size=6, seed=7)
    case = next(c for c in m.full if c.issue_number == 42)
    assert case.changed_files == ("src/fix.ts",)


def test_manifest_pins_the_base_commit_for_every_case():
    issues, prs = build(5)
    m = build_manifest(issues, prs, repo=REPO, limit=LIMIT, dev_size=1, full_size=2, seed=7)
    for case in m.full:
        assert case.base_commit == prs[case.pr_number].base_commit
    parsed = json.loads(manifest_to_json(m))
    for entry in parsed["full"] + parsed["dev"]:
        assert len(entry["base_commit"]) == 40


# --- provenance and sizing (IMPORTANT 2 and 3) ---------------------------


def test_undersized_eligible_pool_is_an_error_not_a_short_manifest():
    """A 300-case run must not be publishable as if it were the 500-case
    run the plan promises."""
    issues, prs = build(10)
    with pytest.raises(ManifestSizeError, match="only 10 eligible"):
        build_manifest(
            issues, prs, repo=REPO, limit=LIMIT,
            dev_size=2, full_size=20, seed=1,
        )


def test_undersized_pool_is_allowed_explicitly_and_recorded():
    issues, prs = build(10)
    m = build_manifest(
        issues, prs, repo=REPO, limit=LIMIT,
        dev_size=2, full_size=20, seed=1, allow_undersized=True,
    )
    assert len(m.full) == 10
    assert m.meta.undersized is True
    assert m.meta.full_size == 20


def test_manifest_records_eligible_and_examined_counts():
    issues, prs = build(10)
    bad_issue, bad_pr = make_pair(99, labels=("feature-request",))
    issues.append(bad_issue)
    prs[bad_pr.number] = bad_pr

    m = build_manifest(
        issues, prs, repo=REPO, limit=LIMIT, dev_size=2, full_size=5, seed=1
    )
    assert m.meta.total_examined == 11
    assert m.meta.total_eligible == 10
    assert len(m.full) == 5
    assert m.meta.undersized is False


def test_manifest_json_carries_provenance():
    issues, prs = build(10)
    m = build_manifest(
        issues, prs, repo=REPO, limit=LIMIT, dev_size=2, full_size=5, seed=1,
        generated_at="2026-09-21T00:00:00+00:00",
    )
    meta = json.loads(manifest_to_json(m))["meta"]
    assert meta == {
        "dev_size": 2,
        "full_size": 5,
        "generated_at": "2026-09-21T00:00:00+00:00",
        "limit": LIMIT,
        "max_changed_files": MAX_CHANGED_FILES,
        "min_changed_files": MIN_CHANGED_FILES,
        "repo": REPO,
        "seed": 1,
        "total_eligible": 10,
        "total_examined": 10,
        "undersized": False,
    }


def test_manifest_size_error_is_a_value_error():
    assert issubclass(ManifestSizeError, ValueError)

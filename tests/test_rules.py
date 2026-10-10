from miresolver.rules import (
    CandidateIssue,
    CandidatePR,
    is_eligible,
    source_files,
)

BUG = ("bug",)


def issue(**kw) -> CandidateIssue:
    base = dict(number=1, title="crash on empty config",
                body="it crashes", labels=BUG, linked_pr_numbers=(10,))
    base.update(kw)
    return CandidateIssue(**base)


def pr(**kw) -> CandidatePR:
    base = dict(number=10, merged=True, title="fix parser crash",
                changed_files=("src/config/parser.ts",),
                base_commit="a" * 40)
    base.update(kw)
    return CandidatePR(**base)


def test_eligible_when_all_rules_pass():
    result = is_eligible(issue(), {10: pr()})
    assert result.eligible is True
    assert result.reason == "ok"


def test_rejects_issue_without_bug_label():
    result = is_eligible(issue(labels=("feature-request",)), {10: pr()})
    assert result.eligible is False
    assert result.reason == "no_bug_label"


def test_bug_label_match_is_case_insensitive():
    result = is_eligible(issue(labels=("Bug",)), {10: pr()})
    assert result.eligible is True
    assert result.reason == "ok"


def test_rejects_issue_with_two_linked_prs():
    result = is_eligible(
        issue(linked_pr_numbers=(10, 11)),
        {10: pr(), 11: pr(number=11)},
    )
    assert result.reason == "not_exactly_one_merged_pr"


def test_rejects_issue_whose_only_pr_is_unmerged():
    result = is_eligible(issue(), {10: pr(merged=False)})
    assert result.reason == "not_exactly_one_merged_pr"


def test_counts_only_merged_prs_toward_the_one_pr_rule():
    result = is_eligible(
        issue(linked_pr_numbers=(10, 11)),
        {10: pr(), 11: pr(number=11, merged=False)},
    )
    assert result.eligible is True


def test_rejects_pr_touching_zero_files():
    result = is_eligible(issue(), {10: pr(changed_files=())})
    assert result.reason == "file_count_out_of_range"


def test_rejects_pr_touching_six_files():
    files = tuple(f"src/f{i}.ts" for i in range(6))
    result = is_eligible(issue(), {10: pr(changed_files=files)})
    assert result.reason == "file_count_out_of_range"


def test_accepts_pr_touching_five_files():
    files = tuple(f"src/f{i}.ts" for i in range(5))
    result = is_eligible(issue(), {10: pr(changed_files=files)})
    assert result.eligible is True


def test_rejects_revert_pr():
    result = is_eligible(issue(), {10: pr(title='Revert "fix parser crash"')})
    assert result.reason == "revert"


# --- source-file filtering (IMPORTANT 5) ---------------------------------


def test_source_files_keeps_ordinary_source_paths():
    paths = ("src/config/parser.ts", "lib/util.go")
    assert source_files(paths) == paths


def test_source_files_drops_test_paths():
    assert source_files(
        (
            "src/config/parser.ts",
            "src/config/test/parser.test.ts",
            "tests/test_parser.py",
            "src/__tests__/parser.ts",
            "pkg/parser_test.go",
            "spec/parser_spec.rb",
        )
    ) == ("src/config/parser.ts",)


def test_source_files_drops_docs_lockfiles_and_i18n():
    assert source_files(
        (
            "src/config/parser.ts",
            "CHANGELOG.md",
            "docs/usage.rst",
            "package-lock.json",
            "yarn.lock",
            "go.sum",
            "i18n/de/messages.json",
            "src/locales/fr.po",
            "src/__snapshots__/parser.snap",
        )
    ) == ("src/config/parser.ts",)


def test_source_files_is_case_insensitive():
    assert source_files(("Docs/Guide.MD", "Package-Lock.json")) == ()


def test_file_count_rule_ignores_non_source_files():
    files = ("src/a.ts",) + tuple(f"test/t{i}.ts" for i in range(10))
    assert is_eligible(issue(), {10: pr(changed_files=files)}).eligible is True


def test_rejects_pr_with_no_source_files():
    files = ("CHANGELOG.md", "docs/guide.md")
    result = is_eligible(issue(), {10: pr(changed_files=files)})
    assert result.reason == "file_count_out_of_range"


def test_file_count_rule_counts_six_source_files_as_out_of_range():
    files = tuple(f"src/f{i}.ts" for i in range(6)) + ("test/t.ts",)
    result = is_eligible(issue(), {10: pr(changed_files=files)})
    assert result.reason == "file_count_out_of_range"


# --- linkage truncation (IMPORTANT 6) ------------------------------------


def test_rejects_issue_whose_linked_pr_list_was_truncated():
    """A truncated list can hide a second merged PR, which would make an
    ineligible issue look like it has exactly one and admit it with the
    wrong answer key."""
    result = is_eligible(issue(linkage_truncated=True), {10: pr()})
    assert result.eligible is False
    assert result.reason == "linkage_truncated"


def test_rejects_pr_whose_changed_file_list_was_truncated():
    """A truncated file list is an incomplete answer key, and can also make
    a sprawling PR look like a small one."""
    result = is_eligible(issue(), {10: pr(files_truncated=True)})
    assert result.eligible is False
    assert result.reason == "file_list_truncated"


# --- base commit (CRITICAL 1) --------------------------------------------


def test_rejects_merged_pr_without_a_base_commit():
    """Without a pre-fix commit there is nothing to index, so the case
    cannot be evaluated under the leakage control."""
    result = is_eligible(issue(), {10: pr(base_commit="")})
    assert result.eligible is False
    assert result.reason == "missing_base_commit"


# --- linkage counting edge cases -----------------------------------------


def test_rejects_issue_with_no_linked_prs():
    """The commonest real shape: a closed issue nobody fixed with a PR."""
    result = is_eligible(issue(linked_pr_numbers=()), {})
    assert result.eligible is False
    assert result.reason == "not_exactly_one_merged_pr"


def test_linked_pr_number_missing_from_the_pr_mapping_is_ignored():
    """A linked number we never fetched a PR record for cannot be counted
    or used as an answer key, so it must not reach prs[n]."""
    result = is_eligible(issue(linked_pr_numbers=(10, 77)), {10: pr()})
    assert result.eligible is True
    assert result.reason == "ok"


def test_issue_whose_only_linked_pr_is_missing_is_rejected():
    result = is_eligible(issue(linked_pr_numbers=(77,)), {10: pr()})
    assert result.reason == "not_exactly_one_merged_pr"

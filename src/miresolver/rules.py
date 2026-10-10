from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MIN_CHANGED_FILES = 1
MAX_CHANGED_FILES = 5
BUG_LABELS = frozenset({"bug"})

# A pull request's changed files become the answer key, so anything that is
# not production source has to be stripped before counting and before
# publishing. Leaving a test file in the key lets a retrieval system score a
# hit for naming the test that came with the fix, which inflates the headline
# number. These patterns are documented in the README as part of the
# inclusion rules; change one and change the other.
EXCLUDED_PATH_SEGMENTS = frozenset(
    {
        "test",
        "tests",
        "testing",
        "__tests__",
        "__mocks__",
        "__snapshots__",
        "spec",
        "specs",
        "fixtures",
        "doc",
        "docs",
        "i18n",
        "locale",
        "locales",
        "translations",
    }
)
EXCLUDED_FILENAMES = frozenset(
    {
        "cargo.lock",
        "composer.lock",
        "gemfile.lock",
        "go.sum",
        "npm-shrinkwrap.json",
        "package-lock.json",
        "packages.lock.json",
        "pdm.lock",
        "pnpm-lock.yaml",
        "podfile.lock",
        "poetry.lock",
        "uv.lock",
        "yarn.lock",
    }
)
EXCLUDED_SUFFIXES = (".md", ".rst", ".txt", ".po", ".pot", ".snap")
TEST_FILENAME_MARKERS = (".test.", ".spec.", "_test.", "_spec.")
TEST_FILENAME_PREFIXES = ("test_", "test-")


def _is_source_path(path: str) -> bool:
    lowered = path.lower()
    parts = lowered.split("/")
    filename = parts[-1]

    if set(parts[:-1]) & EXCLUDED_PATH_SEGMENTS:
        return False
    if filename in EXCLUDED_FILENAMES:
        return False
    if lowered.endswith(EXCLUDED_SUFFIXES):
        return False
    if any(marker in filename for marker in TEST_FILENAME_MARKERS):
        return False
    if filename.startswith(TEST_FILENAME_PREFIXES):
        return False
    return True


def source_files(paths: Sequence[str]) -> tuple[str, ...]:
    """Keep only production source paths, preserving order.

    Used both to count files for the inclusion rule and to build the answer
    key, so the two can never disagree.
    """
    return tuple(p for p in paths if _is_source_path(p))


@dataclass(frozen=True)
class CandidateIssue:
    number: int
    title: str
    body: str
    labels: tuple[str, ...]
    linked_pr_numbers: tuple[int, ...]
    #: True when GitHub returned more linked pull requests than one page
    #: held, so linked_pr_numbers may be missing a second merged PR.
    linkage_truncated: bool = False


@dataclass(frozen=True)
class CandidatePR:
    number: int
    merged: bool
    title: str
    changed_files: tuple[str, ...]
    #: The commit the repository must be indexed at to evaluate this case:
    #: the base branch immediately before the fix landed. Empty when GitHub
    #: could not report one, which disqualifies the case.
    base_commit: str = ""
    #: True when the pull request touched more files than one page held, so
    #: changed_files is an incomplete answer key.
    files_truncated: bool = False


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reason: str


def _is_revert(pr: CandidatePR) -> bool:
    return pr.title.strip().lower().startswith("revert")


def is_eligible(
    issue: CandidateIssue, prs: Mapping[int, CandidatePR]
) -> Eligibility:
    if not BUG_LABELS.intersection(label.lower() for label in issue.labels):
        return Eligibility(False, "no_bug_label")

    if issue.linkage_truncated:
        return Eligibility(False, "linkage_truncated")

    merged = [
        prs[n] for n in issue.linked_pr_numbers if n in prs and prs[n].merged
    ]
    if len(merged) != 1:
        return Eligibility(False, "not_exactly_one_merged_pr")

    pr = merged[0]
    if _is_revert(pr):
        return Eligibility(False, "revert")

    if not pr.base_commit:
        return Eligibility(False, "missing_base_commit")

    if pr.files_truncated:
        return Eligibility(False, "file_list_truncated")

    n_source = len(source_files(pr.changed_files))
    if not MIN_CHANGED_FILES <= n_source <= MAX_CHANGED_FILES:
        return Eligibility(False, "file_count_out_of_range")

    return Eligibility(True, "ok")

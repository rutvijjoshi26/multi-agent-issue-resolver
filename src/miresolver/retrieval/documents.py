from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from miresolver.rules import CandidateIssue

MAX_EMBED_CHARS = 8000


@dataclass(frozen=True)
class IssueDocument:
    issue_number: int
    embed_text: str
    payload: Mapping[str, object]


def build_document(
    issue: CandidateIssue,
    *,
    pr_number: int | None = None,
    base_commit: str = "",
    changed_files: Sequence[str] = (),
) -> IssueDocument:
    """Turn an issue into one indexable document.

    Only symptom text is embedded. The resolution rides along as payload so
    that a new symptom matches old symptoms rather than old solutions.
    """
    embed_text = f"{issue.title}\n\n{issue.body}".strip()[:MAX_EMBED_CHARS]
    payload: dict[str, object] = {
        "issue_number": issue.number,
        "title": issue.title,
        "body": issue.body[:MAX_EMBED_CHARS],
        "labels": tuple(issue.labels),
        "pr_number": pr_number,
        "base_commit": base_commit,
        "changed_files": tuple(changed_files),
    }
    return IssueDocument(
        issue_number=issue.number,
        embed_text=embed_text,
        payload=MappingProxyType(payload),
    )

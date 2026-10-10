"""Prompt text for the three agent tiers.

Kept apart from the agent logic so prompt changes show up as one-file diffs
that can be compared across evaluation runs.
"""

from collections.abc import Sequence

from miresolver.agents.types import Candidate, NewIssue, Verdict

ISSUE_BODY_CHARS = 4000

SCREEN_SYSTEM = (
    "You triage bug reports for a large software repository. You are shown "
    "a new bug report and a list of previously solved issues. Score how "
    "likely each past issue shares the new report's root cause, from 0 "
    "(unrelated) to 1 (almost certainly the same bug). Judge the underlying "
    "cause, not surface wording. Reply with JSON only, shaped as "
    '{"scores": [{"issue": <number>, "score": <0..1>}]}, with one entry per '
    "past issue."
)

INVESTIGATE_SYSTEM = (
    "You are a senior engineer investigating whether a past fix is a useful "
    "precedent for a new bug. You are given the new bug report, one past "
    "issue with the files its fixing pull request changed, and the current "
    "content of those files at the commit the new bug is reported against. "
    "Decide whether the precedent applies to the new bug, which files in "
    "the current code would most likely need to change, and what approach "
    "to take. Only name files that exist in the repository; you may name "
    "files beyond the precedent's if the code shows the logic has moved. Be "
    "calibrated: a confidence of 0.5 should mean you expect to be right "
    "half the time. Reply with JSON only, shaped as "
    '{"applies": <bool>, "confidence": <0..1>, "files": [<path>, ...], '
    '"approach": <string>, "reasoning": <string>}.'
)

SYNTHESIZE_SYSTEM = (
    "You combine investigator reports into one recommendation for fixing a "
    "new bug. Weigh each report by its confidence and evidence, merge "
    "overlapping file suggestions, and rank files with the most likely "
    "first. Cite only the past issues whose reports you relied on. If the "
    "reports do not support a confident recommendation, give a low "
    "confidence rather than guessing. Reply with JSON only, shaped as "
    '{"approach": <string>, "files": [<path>, ...], '
    '"cited_issues": [<number>, ...], "confidence": <0..1>}.'
)


def _issue_block(issue: NewIssue) -> str:
    return f"Title: {issue.title}\n\n{issue.body[:ISSUE_BODY_CHARS]}"


def screen_prompt(
    issue: NewIssue, candidates: Sequence[Candidate], body_chars: int
) -> str:
    lines = ["NEW BUG REPORT", _issue_block(issue), "", "PAST SOLVED ISSUES"]
    for c in candidates:
        lines.append(f"#{c.issue_number}: {c.title}")
        lines.append(c.body[:body_chars].replace("\n", " "))
        lines.append("")
    return "\n".join(lines)


def investigate_prompt(
    issue: NewIssue, candidate: Candidate, file_sections: Sequence[str]
) -> str:
    return "\n".join(
        [
            "NEW BUG REPORT",
            _issue_block(issue),
            "",
            f"PAST ISSUE #{candidate.issue_number}: {candidate.title}",
            candidate.body[:ISSUE_BODY_CHARS],
            "",
            f"It was fixed by pull request {candidate.pr_number}, which changed: "
            + ", ".join(candidate.changed_files),
            "",
            f"CURRENT CODE AT COMMIT {issue.base_commit}",
            *file_sections,
        ]
    )


def synthesize_prompt(issue: NewIssue, verdicts: Sequence[Verdict]) -> str:
    lines = ["NEW BUG REPORT", _issue_block(issue), "", "INVESTIGATOR REPORTS"]
    for v in verdicts:
        lines += [
            f"Precedent #{v.issue_number} (fixed by PR {v.pr_number}): "
            f"applies={v.applies}, confidence={v.confidence:.2f}",
            "Files: " + (", ".join(v.files) or "(no files named)"),
            f"Approach: {v.approach}",
            f"Reasoning: {v.reasoning}",
            "",
        ]
    return "\n".join(lines)

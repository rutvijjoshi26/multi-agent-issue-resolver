from miresolver.agents.parsing import clamp_unit, clean_paths
from miresolver.agents.prompts import INVESTIGATE_SYSTEM, investigate_prompt
from miresolver.agents.types import Candidate, NewIssue, ResolverOptions, Verdict
from miresolver.code.reader import CodeReader, CodeReadError
from miresolver.models.chat import ChatClient, ChatError, Usage, extract_json

INVESTIGATE_MAX_TOKENS = 900


async def _file_sections(
    issue: NewIssue, candidate: Candidate, reader: CodeReader, options: ResolverOptions
) -> list[str]:
    sections: list[str] = []
    for path in candidate.changed_files[: options.max_files_per_candidate]:
        try:
            text = await reader.read(issue.base_commit, path)
        except CodeReadError as exc:
            sections.append(f"--- {path}: could not be read ({exc})")
            continue
        if text is None:
            sections.append(
                f"--- {path}: does not exist at this commit (it may have been "
                f"moved, renamed, or added by the past fix)"
            )
            continue
        budget = options.file_char_budget
        note = f" (truncated to the first {budget} characters)" if len(text) > budget else ""
        sections.append(f"--- {path}{note}\n{text[:budget]}")
    return sections


def _verdict(candidate: Candidate, text: str) -> Verdict:
    data = extract_json(text)
    try:
        confidence = clamp_unit(data.get("confidence"))
    except ValueError as exc:
        raise ChatError(f"verdict confidence invalid: {exc}") from exc
    return Verdict(
        issue_number=candidate.issue_number,
        pr_number=candidate.pr_number,
        applies=data.get("applies") is True,
        confidence=confidence,
        files=clean_paths(data.get("files")),
        approach=str(data.get("approach") or ""),
        reasoning=str(data.get("reasoning") or ""),
    )


async def investigate(
    issue: NewIssue,
    candidate: Candidate,
    chat: ChatClient,
    reader: CodeReader,
    options: ResolverOptions,
) -> tuple[Verdict, Usage]:
    """Judge one precedent against the code at the new issue's base commit.

    Never raises for model trouble: a failed investigator becomes a failed
    verdict, so the synthesizer works with whatever the others found.
    """
    sections = await _file_sections(issue, candidate, reader, options)
    prompt = investigate_prompt(issue, candidate, sections)
    try:
        result = await chat.complete(
            INVESTIGATE_SYSTEM,
            [{"role": "user", "content": prompt}],
            max_tokens=INVESTIGATE_MAX_TOKENS,
            json_mode=True,
        )
    except ChatError as exc:
        return Verdict.failure(candidate, f"model call failed: {exc}"), Usage()
    try:
        return _verdict(candidate, result.text), result.usage
    except ChatError as exc:
        return Verdict.failure(candidate, f"unusable JSON verdict: {exc}"), result.usage

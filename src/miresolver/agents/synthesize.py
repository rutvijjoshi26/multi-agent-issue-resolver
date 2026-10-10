from collections.abc import Sequence
from dataclasses import dataclass

from miresolver.agents.parsing import clamp_unit, clean_paths
from miresolver.agents.prompts import SYNTHESIZE_SYSTEM, synthesize_prompt
from miresolver.agents.types import Citation, NewIssue, Verdict
from miresolver.models.chat import ChatClient, ChatError, Usage, extract_json

SYNTHESIZE_MAX_TOKENS = 900


@dataclass(frozen=True)
class SynthesisResult:
    approach: str
    files: tuple[str, ...]
    citations: tuple[Citation, ...]
    confidence: float
    abstained: bool
    reason: str | None
    usage: Usage
    #: True when the synthesis call failed and the answer was merged from
    #: the verdicts directly.
    fallback: bool = False


def merge_verdict_files(verdicts: Sequence[Verdict]) -> list[str]:
    """Rank files by the summed confidence of the verdicts naming them.

    Ties keep first-appearance order, so the result is deterministic.
    """
    weight: dict[str, float] = {}
    for v in verdicts:
        for path in v.files:
            weight[path] = weight.get(path, 0.0) + v.confidence
    order = {path: i for i, path in enumerate(weight)}
    return sorted(weight, key=lambda p: (-weight[p], order[p]))


def _cite(verdicts: Sequence[Verdict]) -> tuple[Citation, ...]:
    return tuple(Citation(v.issue_number, v.pr_number) for v in verdicts)


def _would_be_files(usable: Sequence[Verdict]) -> tuple[str, ...]:
    applying = [v for v in usable if v.applies]
    others = [v for v in usable if not v.applies]
    files = merge_verdict_files(applying)
    files += [p for p in merge_verdict_files(others) if p not in files]
    return tuple(files)


async def synthesize(
    issue: NewIssue,
    verdicts: Sequence[Verdict],
    chat: ChatClient,
    *,
    threshold: float,
) -> SynthesisResult:
    """Merge investigator verdicts into one answer, or abstain.

    Abstention is decided by fixed rules over the verdicts and the
    synthesizer's stated confidence, so a model that is overconfident in
    its prose still cannot answer without a confident precedent behind it.
    """
    usable = [v for v in verdicts if not v.failed]
    if not usable:
        return SynthesisResult("", (), (), 0.0, True, "no_verdicts", Usage())

    confident = sorted(
        (v for v in usable if v.applies and v.confidence >= threshold),
        key=lambda v: -v.confidence,
    )
    if not confident:
        best = max(usable, key=lambda v: (v.applies, v.confidence))
        return SynthesisResult(
            approach=best.approach,
            files=_would_be_files(usable),
            citations=_cite([v for v in usable if v.applies]),
            confidence=max((v.confidence for v in usable if v.applies), default=0.0),
            abstained=True,
            reason="no_confident_precedent",
            usage=Usage(),
        )

    strongest = confident[0]
    fallback_files = tuple(merge_verdict_files([v for v in usable if v.applies]))
    usage = Usage()
    try:
        result = await chat.complete(
            SYNTHESIZE_SYSTEM,
            [{"role": "user", "content": synthesize_prompt(issue, usable)}],
            max_tokens=SYNTHESIZE_MAX_TOKENS,
            json_mode=True,
        )
        usage = result.usage
        data = extract_json(result.text)
        try:
            confidence = clamp_unit(data.get("confidence"))
        except ValueError as exc:
            raise ChatError(f"synthesis confidence invalid: {exc}") from exc
    except Exception:  # any synthesis failure falls back to the verdicts
        return SynthesisResult(
            approach=strongest.approach,
            files=tuple(merge_verdict_files(confident)),
            citations=_cite(confident),
            confidence=strongest.confidence,
            abstained=False,
            reason=None,
            usage=usage,
            fallback=True,
        )

    by_number = {v.issue_number: v for v in usable}
    cited: list[Verdict] = []
    raw_cited = data.get("cited_issues")
    for raw in raw_cited if isinstance(raw_cited, list) else []:
        try:
            v = by_number.get(int(raw))
        except (TypeError, ValueError, OverflowError):
            continue
        if v is not None and v not in cited:
            cited.append(v)

    low = confidence < threshold
    return SynthesisResult(
        approach=str(data.get("approach") or strongest.approach),
        files=clean_paths(data.get("files")) or fallback_files,
        citations=_cite(cited or confident),
        confidence=confidence,
        abstained=low,
        reason="low_synthesis_confidence" if low else None,
        usage=usage,
    )

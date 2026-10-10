from collections.abc import Sequence
from dataclasses import replace

from miresolver.agents.parsing import clamp_unit
from miresolver.agents.prompts import SCREEN_SYSTEM, screen_prompt
from miresolver.agents.types import Candidate, NewIssue
from miresolver.models.chat import ChatClient, ChatError, Usage, extract_json

SCREEN_MAX_TOKENS = 600


def _scores(text: str) -> dict[int, float]:
    entries = extract_json(text).get("scores")
    if not isinstance(entries, list):
        raise ChatError("screening reply had no scores list")
    scores: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            scores[int(entry["issue"])] = clamp_unit(entry["score"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return scores


async def screen(
    issue: NewIssue,
    candidates: Sequence[Candidate],
    chat: ChatClient,
    *,
    top_n: int,
    floor: float,
    body_chars: int = 600,
) -> tuple[list[Candidate], Usage]:
    """Score every candidate in one cheap call and keep the best top_n.

    If screening fails, the candidates are kept in retrieval order instead:
    an outage in the cheapest tier should cost quality, not the answer.
    """
    if not candidates:
        return [], Usage()
    usage = Usage()
    try:
        result = await chat.complete(
            SCREEN_SYSTEM,
            [{"role": "user", "content": screen_prompt(issue, candidates, body_chars)}],
            max_tokens=SCREEN_MAX_TOKENS,
            json_mode=True,
        )
        usage = result.usage
        scores = _scores(result.text)
    except Exception:  # any screening failure degrades to retrieval order
        ordered = sorted(candidates, key=lambda c: c.retrieval_rank)
        return list(ordered[:top_n]), usage

    scored = [replace(c, screen_score=scores.get(c.issue_number, 0.0)) for c in candidates]
    kept = [c for c in scored if c.screen_score >= floor]
    kept.sort(key=lambda c: (-c.screen_score, c.retrieval_rank))
    return kept[:top_n], usage

"""Scoring predicted files against the answer key.

Every rate is a plain fraction of cases, rounded to four places, and is
null when its denominator is zero rather than a misleading 0.
"""

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

TOP_K = 3


@dataclass(frozen=True)
class CaseScore:
    hit_at_1: bool
    hit_at_3: bool
    precision_at_3: float
    recall_at_3: float
    abstained: bool
    #: Some retrieved candidate's fix touched an answer-key file, so a
    #: correct answer was reachable from the evidence retrieval supplied.
    precedent_available: bool

    @property
    def answered(self) -> bool:
        return not self.abstained

    @property
    def answered_correct(self) -> bool:
        return self.answered and self.hit_at_3

    @property
    def confident_wrong(self) -> bool:
        return self.answered and not self.hit_at_3

    @property
    def correct_abstention(self) -> bool:
        return self.abstained and not self.precedent_available


def _norm(path: str) -> str:
    return path.strip().removeprefix("./")


def score_case(
    predicted: Sequence[str],
    answer: Sequence[str],
    *,
    abstained: bool,
    candidate_files: Iterable[str],
) -> CaseScore:
    """Score one case. `predicted` is ranked, most likely first.

    Scored on the would-be suggestion even when the system abstained, so
    abstention can be judged as a decision separately from file accuracy.
    """
    key = {_norm(p) for p in answer}
    ranked = list(dict.fromkeys(_norm(p) for p in predicted))
    top = ranked[:TOP_K]
    found = len(key.intersection(top))
    return CaseScore(
        hit_at_1=bool(ranked) and ranked[0] in key,
        hit_at_3=found > 0,
        precision_at_3=found / len(top) if top else 0.0,
        recall_at_3=found / len(key) if key else 0.0,
        abstained=abstained,
        precedent_available=bool(key.intersection(_norm(p) for p in candidate_files)),
    )


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Nearest-rank percentile, so the result is always an observed value."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _file_metrics(scores: Sequence[CaseScore]) -> dict:
    n = len(scores)
    return {
        "cases": n,
        "hit_at_1": _rate(sum(s.hit_at_1 for s in scores), n),
        "hit_at_3": _rate(sum(s.hit_at_3 for s in scores), n),
        "precision_at_3": _mean([s.precision_at_3 for s in scores]),
        "recall_at_3": _mean([s.recall_at_3 for s in scores]),
    }


def summarize(scores: Sequence[CaseScore], latencies: Sequence[float]) -> dict:
    n = len(scores)
    answered = [s for s in scores if s.answered]
    abstained = [s for s in scores if s.abstained]
    median = statistics.median(latencies) if latencies else None
    p95 = percentile(latencies, 95)
    return {
        "cases": n,
        "all_cases": _file_metrics(scores),
        "answered": _file_metrics(answered),
        "abstention_rate": _rate(len(abstained), n),
        "accuracy_when_answering": _rate(
            sum(s.answered_correct for s in answered), len(answered)
        ),
        "confident_wrong_rate": _rate(sum(s.confident_wrong for s in scores), n),
        "precedent_available_rate": _rate(
            sum(s.precedent_available for s in scores), n
        ),
        "correct_abstention_rate": _rate(
            sum(s.correct_abstention for s in abstained), len(abstained)
        ),
        "latency_s": {
            "median": round(median, 3) if median is not None else None,
            "p95": round(p95, 3) if p95 is not None else None,
        },
    }

"""Choose abstention thresholds from an always-answer run.

An always-answer run consults the synthesizer for every case and records
each investigator's verdict and the synthesizer's confidence, so the
abstaining rule can be replayed offline at any pair of thresholds without
another model call. The objective is fixed before looking at any run:
maximise answered-correct minus answered-wrong (answer only where the
system is right more often than wrong), breaking ties toward answering
more cases and then toward lower thresholds. It must only ever be applied
to a dev-set run.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from miresolver.agents.synthesize import AGREEMENT_DEPTH

DEFAULT_GRID = tuple(round(0.05 * i, 2) for i in range(4, 20))  # 0.20 .. 0.95
#: With three investigators, agreement ranges from 0 to 3.
AGREEMENT_LEVELS = (0, 1, 2, 3)


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class CaseSignal:
    hit_at_3: bool
    best_applicable_verdict: float | None
    synthesis_confidence: float
    agreement: int = 0


def _agreement(rec: Mapping) -> int:
    """Recorded agreement, or recomputed for runs made before it was recorded."""
    if "agreement" in rec:
        return int(rec["agreement"] or 0)
    tops = [
        set((v.get("files") or [])[:AGREEMENT_DEPTH])
        for v in rec.get("verdicts") or []
        if not v.get("error")
    ]
    files = (rec.get("predicted_files") or [])[:AGREEMENT_DEPTH]
    return max((sum(f in top for top in tops) for f in files), default=0)


def load_signals(records: Iterable[Mapping]) -> list[CaseSignal]:
    signals = []
    for rec in records:
        if rec.get("status") != "scored":
            continue
        applicable = [
            float(v["confidence"])
            for v in rec.get("verdicts") or []
            if v.get("applies") and not v.get("error")
        ]
        signals.append(
            CaseSignal(
                hit_at_3=bool(rec.get("hit_at_3")),
                best_applicable_verdict=max(applicable) if applicable else None,
                synthesis_confidence=float(rec.get("confidence") or 0.0),
                agreement=_agreement(rec),
            )
        )
    return signals


def load_run(run_dir: Path) -> tuple[dict, list[CaseSignal]]:
    summary = json.loads((run_dir / "summary.json").read_text())
    lines = (run_dir / "cases.jsonl").read_text().splitlines()
    return summary, load_signals(json.loads(line) for line in lines if line.strip())


def would_answer(
    signal: CaseSignal,
    *,
    verdict_threshold: float,
    synthesis_threshold: float,
    min_agreement: int = 0,
) -> bool:
    return (
        signal.best_applicable_verdict is not None
        and signal.best_applicable_verdict >= verdict_threshold
        and signal.synthesis_confidence >= synthesis_threshold
        and signal.agreement >= min_agreement
    )


def evaluate_thresholds(
    signals: Sequence[CaseSignal],
    verdict_threshold: float,
    synthesis_threshold: float,
    min_agreement: int = 0,
) -> dict:
    answered = [
        s for s in signals
        if would_answer(s, verdict_threshold=verdict_threshold,
                        synthesis_threshold=synthesis_threshold,
                        min_agreement=min_agreement)
    ]
    right = sum(s.hit_at_3 for s in answered)
    n = len(signals)
    return {
        "verdict_threshold": verdict_threshold,
        "synthesis_threshold": synthesis_threshold,
        "min_agreement": min_agreement,
        "cases": n,
        "answered": len(answered),
        "answered_correct": right,
        "answered_wrong": len(answered) - right,
        "net": right - (len(answered) - right),
        "abstention_rate": round(1 - len(answered) / n, 4) if n else None,
        "accuracy_when_answering": round(right / len(answered), 4) if answered else None,
        "confident_wrong_rate": round((len(answered) - right) / n, 4) if n else None,
    }


def choose_thresholds(
    signals: Sequence[CaseSignal],
    *,
    grid: Sequence[float] = DEFAULT_GRID,
    agreement_levels: Sequence[int] = AGREEMENT_LEVELS,
) -> dict:
    if not signals:
        raise CalibrationError("no scored cases to calibrate on")
    options = [
        evaluate_thresholds(signals, tv, ts, ma)
        for tv in grid for ts in grid for ma in agreement_levels
    ]
    options.sort(
        key=lambda o: (
            -o["net"], -o["answered"], o["min_agreement"],
            o["verdict_threshold"], o["synthesis_threshold"],
        )
    )
    return options[0]

import json
from pathlib import Path

import pytest

from miresolver.evaluation.metrics import percentile, score_case, summarize

GOLDEN = Path(__file__).parent / "golden_summary.json"


def test_hit_at_rank_two_counts_for_top_three_not_top_one():
    s = score_case(["x.ts", "a.ts", "y.ts", "b.ts"], ["a.ts", "b.ts"],
                   abstained=False, candidate_files=[])
    assert (s.hit_at_1, s.hit_at_3) == (False, True)
    assert s.precision_at_3 == pytest.approx(1 / 3)
    assert s.recall_at_3 == pytest.approx(1 / 2)
    assert s.answered_correct and not s.confident_wrong


def test_empty_prediction_scores_zero_without_dividing_by_zero():
    s = score_case([], ["a.ts"], abstained=False, candidate_files=[])
    assert (s.hit_at_3, s.precision_at_3, s.recall_at_3) == (False, 0.0, 0.0)
    assert s.confident_wrong


def test_leading_dot_slash_is_ignored_when_matching():
    s = score_case(["./a.ts"], ["a.ts"], abstained=False, candidate_files=[])
    assert s.hit_at_1


def test_abstaining_is_never_confidently_wrong():
    s = score_case(["x.ts"], ["a.ts"], abstained=True, candidate_files=["q.ts"])
    assert not s.confident_wrong and not s.answered_correct
    assert not s.precedent_available
    assert s.correct_abstention


def test_precedent_available_when_any_candidate_fix_touched_an_answer_file():
    s = score_case(["x.ts"], ["a.ts"], abstained=True, candidate_files=["q.ts", "a.ts"])
    assert s.precedent_available and not s.correct_abstention


def test_percentile_uses_nearest_rank():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([], 50) is None


def _fixture_scores():
    return [
        score_case(["a.ts"], ["a.ts"], abstained=False, candidate_files=["a.ts"]),
        score_case(["x.ts", "b.ts"], ["b.ts"], abstained=False, candidate_files=["b.ts"]),
        score_case(["x.ts"], ["c.ts"], abstained=False, candidate_files=[]),
        score_case(["d.ts"], ["d.ts"], abstained=True, candidate_files=["d.ts"]),
        score_case([], ["e.ts"], abstained=True, candidate_files=[]),
    ]


def test_summary_matches_hand_computed_values():
    summary = summarize(_fixture_scores(), [1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["cases"] == 5
    assert summary["abstention_rate"] == 0.4
    assert summary["accuracy_when_answering"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["confident_wrong_rate"] == 0.2
    assert summary["all_cases"]["hit_at_3"] == 0.6
    assert summary["all_cases"]["hit_at_1"] == 0.4
    assert summary["answered"]["cases"] == 3
    assert summary["correct_abstention_rate"] == 0.5
    assert summary["precedent_available_rate"] == 0.6
    assert summary["latency_s"] == {"median": 3.0, "p95": 5.0}


def test_summary_golden_file():
    summary = summarize(_fixture_scores(), [1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary == json.loads(GOLDEN.read_text())


def test_summary_of_nothing_has_nulls_not_errors():
    summary = summarize([], [])
    assert summary["cases"] == 0
    assert summary["accuracy_when_answering"] is None
    assert summary["latency_s"] == {"median": None, "p95": None}


def test_summary_splits_cases_by_whether_a_precedent_was_retrieved():
    summary = summarize(_fixture_scores(), [1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["with_precedent"] == {
        "cases": 3, "hit_at_3": 1.0, "abstention_rate": 0.3333,
    }
    assert summary["without_precedent"] == {
        "cases": 2, "hit_at_3": 0.0, "abstention_rate": 0.5,
    }

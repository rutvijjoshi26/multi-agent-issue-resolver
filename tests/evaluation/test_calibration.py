import json

import pytest

from miresolver.evaluation.calibration import (
    CalibrationError,
    choose_thresholds,
    load_signals,
    would_answer,
)


def case(hit, verdicts, synth, status="scored"):
    return {"status": status, "hit_at_3": hit, "confidence": synth,
            "verdicts": [{"applies": a, "confidence": c, "error": None} for a, c in verdicts]}


def test_would_answer_needs_a_confident_applicable_verdict_and_synthesis():
    signal = load_signals([case(True, [(True, 0.7), (False, 0.95)], 0.6)])[0]
    assert would_answer(signal, verdict_threshold=0.7, synthesis_threshold=0.6)
    assert not would_answer(signal, verdict_threshold=0.75, synthesis_threshold=0.6)
    assert not would_answer(signal, verdict_threshold=0.7, synthesis_threshold=0.65)


def test_failed_verdicts_do_not_count():
    rec = case(True, [], 0.9)
    rec["verdicts"] = [{"applies": True, "confidence": 0.9, "error": "timeout"}]
    signal = load_signals([rec])[0]
    assert not would_answer(signal, verdict_threshold=0.1, synthesis_threshold=0.1)


def test_unscored_cases_are_ignored():
    assert load_signals([case(True, [(True, 0.9)], 0.9, status="error")]) == []


def test_choice_maximises_correct_minus_wrong_answers():
    records = (
        [case(True, [(True, 0.9)], 0.9)] * 3
        + [case(False, [(True, 0.6)], 0.9)] * 4
        + [case(True, [(True, 0.6)], 0.9)] * 1
    )
    choice = choose_thresholds(load_signals(records), grid=[0.5, 0.7])
    assert (choice["verdict_threshold"], choice["synthesis_threshold"]) == (0.7, 0.5)
    assert choice["answered"] == 3 and choice["answered_correct"] == 3


def test_ties_prefer_more_answers_then_lower_thresholds():
    records = [case(True, [(True, 0.9)], 0.9), case(False, [(True, 0.6)], 0.6),
               case(True, [(True, 0.6)], 0.6)]
    choice = choose_thresholds(load_signals(records), grid=[0.5, 0.7])
    # (0.5, 0.5): 2 right - 1 wrong = 1, three answers; (0.7, *): 1 - 0 = 1, one answer.
    assert (choice["verdict_threshold"], choice["synthesis_threshold"]) == (0.5, 0.5)


def test_an_empty_run_cannot_be_calibrated():
    with pytest.raises(CalibrationError):
        choose_thresholds([], grid=[0.5])


def write_run(tmp_path, *, set_name="dev", mode="always-answer"):
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"run": {"set": set_name, "mode": mode}}))
    (run / "cases.jsonl").write_text("".join(
        json.dumps(case(h, [(True, c)], c)) + "\n"
        for h, c in [(True, 0.9), (True, 0.8), (False, 0.4)]
    ))
    return run


def test_cli_calibrate_prints_the_chosen_thresholds(tmp_path, capsys):
    from miresolver.cli import main

    assert main(["calibrate", str(write_run(tmp_path))]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["chosen"]["answered"] == 2
    assert out["chosen"]["answered_wrong"] == 0


def test_cli_calibrate_refuses_anything_but_a_dev_run(tmp_path, capsys):
    from miresolver.cli import main

    assert main(["calibrate", str(write_run(tmp_path, set_name="test"))]) == 1
    assert "dev" in capsys.readouterr().err


def test_cli_calibrate_needs_an_always_answer_run(tmp_path, capsys):
    from miresolver.cli import main

    assert main(["calibrate", str(write_run(tmp_path, mode="abstain"))]) == 1
    assert "always-answer" in capsys.readouterr().err

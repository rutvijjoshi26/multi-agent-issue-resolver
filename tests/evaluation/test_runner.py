import asyncio
import json

import pytest

from miresolver.agents.types import Candidate, Citation, Resolution, ResolverOptions
from miresolver.evaluation.runner import (
    EvaluationError,
    estimate_cost,
    leakage_filter,
    run_evaluation,
    select_cases,
)
from miresolver.manifest import EvalCase
from miresolver.models.chat import Usage
from miresolver.models.settings import load_model_settings

MANIFEST = {
    "dev": [{"issue_number": n, "pr_number": n + 1000, "base_commit": "a" * 40,
             "changed_files": [f"src/{n}.ts"]} for n in range(1, 21)],
    "full": [{"issue_number": n, "pr_number": n + 1000, "base_commit": "a" * 40,
              "changed_files": [f"src/{n}.ts"]} for n in range(1, 41)],
    "meta": {"generated_at": "2026-09-21T00:00:00+00:00", "seed": 1},
}

SETTINGS = load_model_settings("""
tiers:
  embed: {provider: openai-compatible, base_url: "http://x/v1", model: emb, api_key_env: K,
          input_cost_per_mtok: 0.02, output_cost_per_mtok: 0}
  screen: {provider: openai-compatible, base_url: "http://x/v1", model: small, api_key_env: K,
           input_cost_per_mtok: 1.0, output_cost_per_mtok: 2.0}
  investigate: {provider: openai-compatible, base_url: "http://x/v1", model: mid, api_key_env: K,
                input_cost_per_mtok: 1.0, output_cost_per_mtok: 2.0}
  synthesize: {provider: openai-compatible, base_url: "http://x/v1", model: big, api_key_env: K}
""")


def test_dev_selection_without_a_limit_is_the_whole_dev_set_in_order():
    cases = select_cases(MANIFEST, which="dev", limit=None, seed=0)
    assert [c.issue_number for c in cases] == list(range(1, 21))
    assert cases[0] == EvalCase(1, 1001, "a" * 40, ("src/1.ts",))


def test_a_limited_selection_is_seeded_and_sorted():
    a = select_cases(MANIFEST, which="dev", limit=5, seed=7)
    b = select_cases(MANIFEST, which="dev", limit=5, seed=7)
    c = select_cases(MANIFEST, which="dev", limit=5, seed=8)
    assert a == b and a != c and len(a) == 5
    assert [x.issue_number for x in a] == sorted(x.issue_number for x in a)


def test_the_full_set_is_refused_unless_explicitly_allowed():
    with pytest.raises(EvaluationError, match="allow"):
        select_cases(MANIFEST, which="full", limit=None, seed=0)
    assert len(select_cases(MANIFEST, which="full", limit=None, seed=0, allow_full=True)) == 40
    with pytest.raises(EvaluationError):
        select_cases(MANIFEST, which="nope", limit=None, seed=0)


def cand(number, pr):
    return Candidate(number, "t", "b", pr, ("x.ts",), 0)


def test_leakage_filter_drops_the_case_its_fix_and_later_fixes():
    exclude = leakage_filter(EvalCase(50, 5000, "a" * 40, ("x.ts",)))
    assert exclude(cand(50, 1))        # the case itself
    assert exclude(cand(7, 5000))      # fixed by the case's own PR
    assert exclude(cand(8, 5001))      # fixed by a later PR
    assert exclude(cand(9, None))      # no known fix, nothing to learn from
    assert not exclude(cand(10, 4999))


def test_estimate_is_an_upper_bound_with_dollars_only_for_priced_tiers():
    est = estimate_cost(10, SETTINGS, ResolverOptions())
    assert est["cases"] == 10
    assert est["tiers"]["synthesize"]["dollars"] is None
    assert est["tiers"]["screen"]["dollars"] > 0
    assert est["tiers"]["investigate"]["input_tokens"] > est["tiers"]["screen"]["input_tokens"]
    assert est["priced_dollars"] == pytest.approx(
        sum(t["dollars"] for t in est["tiers"].values() if t["dollars"] is not None)
    )
    assert est["unpriced_tiers"] == ["synthesize"]


def resolution(files, *, abstained=False, cands=()):
    return Resolution(
        approach="do x", files=tuple(files), citations=(Citation(3, 30),),
        confidence=0.4 if abstained else 0.9, abstained=abstained,
        abstain_reason="low_synthesis_confidence" if abstained else None,
        candidates=tuple(cands),
        usage={"screen": Usage(1000, 100), "investigate": Usage(3000, 300),
               "synthesize": Usage(500, 50)},
    )


def run(cases, resolver, tmp_path, **kwargs):
    texts = kwargs.pop("texts", {c.issue_number: ("title", "body") for c in cases})
    return asyncio.run(run_evaluation(
        cases, issue_text=texts.get, resolver=resolver, settings=SETTINGS,
        out_root=tmp_path, run_meta={"set": "dev", "seed": 3, "limit": None},
        timestamp="20261010T000000Z", **kwargs,
    ))


def test_run_writes_a_summary_and_one_line_per_case(tmp_path):
    cases = select_cases(MANIFEST, which="dev", limit=3, seed=1)
    seen = []

    async def resolver(issue, exclude):
        seen.append((issue.number, issue.base_commit, exclude(cand(issue.number, 1))))
        right = f"src/{issue.number}.ts"
        if issue.number == cases[0].issue_number:
            return resolution(["other.ts", right], cands=[Candidate(1, "", "", 1, (right,), 0)])
        if issue.number == cases[1].issue_number:
            return resolution(["wrong.ts"])
        return resolution([right], abstained=True)

    out = run(cases, resolver, tmp_path)
    assert out.name == "dev-n3-seed3-20261010T000000Z"
    assert all(base == "a" * 40 and excluded for _, base, excluded in seen)

    summary = json.loads((out / "summary.json").read_text())
    lines = [json.loads(l) for l in (out / "cases.jsonl").read_text().splitlines()]
    assert len(lines) == 3
    assert lines[0]["issue_number"] == cases[0].issue_number
    assert lines[0]["hit_at_3"] is True and lines[0]["predicted_files"][1].endswith(".ts")
    assert lines[0]["answer_files"] == list(cases[0].changed_files)
    assert lines[0]["citations"] == [{"issue_number": 3, "pr_number": 30}]
    assert summary["metrics"]["cases"] == 3
    assert summary["metrics"]["abstention_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["metrics"]["accuracy_when_answering"] == 0.5
    assert summary["models"]["screen"] == {"provider": "openai-compatible", "model": "small"}
    assert summary["counts"] == {"selected": 3, "scored": 3, "skipped_no_text": 0,
                                 "errors": 0, "not_run": 0}
    assert summary["tokens"]["investigate"] == {"input_tokens": 9000, "output_tokens": 900}
    # screen: 3 * (1000*1 + 100*2) / 1e6 ; investigate: 3 * (3000 + 600) / 1e6
    assert summary["spend"]["tiers"]["screen"] == pytest.approx(0.0036)
    assert summary["spend"]["tiers"]["synthesize"] is None
    assert summary["spend"]["priced_total"] == pytest.approx(0.0036 + 0.0108)
    assert summary["spend"]["unpriced_tiers"] == ["synthesize"]
    assert summary["stopped_early"] is False
    assert summary["run"]["seed"] == 3


def test_missing_text_is_skipped_and_resolver_errors_are_recorded(tmp_path):
    cases = select_cases(MANIFEST, which="dev", limit=3, seed=1)

    async def resolver(issue, exclude):
        if issue.number == cases[1].issue_number:
            raise RuntimeError("qdrant down")
        return resolution(["x.ts"])

    texts = {cases[1].issue_number: ("t", "b"), cases[2].issue_number: ("t", "b")}
    out = run(cases, resolver, tmp_path, texts=texts)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["counts"] == {"selected": 3, "scored": 1, "skipped_no_text": 1,
                                 "errors": 1, "not_run": 0}
    statuses = [json.loads(l)["status"] for l in (out / "cases.jsonl").read_text().splitlines()]
    assert statuses == ["skipped_no_text", "error", "scored"]
    assert summary["metrics"]["cases"] == 1


def test_the_spend_cap_stops_the_run_early(tmp_path):
    cases = select_cases(MANIFEST, which="dev", limit=10, seed=1)

    async def resolver(issue, exclude):
        return resolution(["x.ts"])

    out = run(cases, resolver, tmp_path, max_cost=0.01)
    summary = json.loads((out / "summary.json").read_text())
    # each case costs 0.0012 + 0.0036 = 0.0048 priced dollars
    assert summary["stopped_early"] is True
    assert summary["counts"]["scored"] == 3
    assert summary["counts"]["not_run"] == 7
    assert summary["spend"]["max_cost"] == 0.01


def test_a_negative_limit_is_rejected():
    with pytest.raises(EvaluationError, match="limit"):
        select_cases(MANIFEST, which="dev", limit=-1, seed=0)


def test_errors_and_unmetered_calls_are_counted_in_the_summary(tmp_path):
    from dataclasses import replace
    cases = select_cases(MANIFEST, which="dev", limit=2, seed=1)

    async def resolver(issue, exclude):
        if issue.number == cases[0].issue_number:
            raise RuntimeError("boom")
        return replace(resolution(["x.ts"]), unmetered_calls=2)

    summary = json.loads((run(cases, resolver, tmp_path) / "summary.json").read_text())
    assert summary["spend"]["unmetered"] == {"error_cases": 1, "model_calls": 2}


def test_each_case_line_records_its_unmetered_calls(tmp_path):
    from dataclasses import replace
    cases = select_cases(MANIFEST, which="dev", limit=1, seed=1)

    async def resolver(issue, exclude):
        return replace(resolution(["x.ts"]), unmetered_calls=1)

    out = run(cases, resolver, tmp_path)
    assert json.loads((out / "cases.jsonl").read_text())["unmetered_calls"] == 1


V2 = {
    "dev": MANIFEST["dev"],
    "test": [{"issue_number": n, "pr_number": n + 1000, "base_commit": "b" * 40,
              "changed_files": [f"src/{n}.ts"]} for n in range(100, 110)],
    "meta": {"version": 2, "seed": 3},
}


def test_the_held_out_test_set_is_refused_unless_this_is_the_final_run():
    with pytest.raises(EvaluationError, match="--final"):
        select_cases(V2, which="test", limit=None, seed=0)
    cases = select_cases(V2, which="test", limit=None, seed=0, allow_test=True)
    assert [c.issue_number for c in cases] == list(range(100, 110))


def test_the_final_run_takes_the_whole_test_set():
    with pytest.raises(EvaluationError, match="whole"):
        select_cases(V2, which="test", limit=5, seed=0, allow_test=True)


def test_a_set_the_manifest_lacks_is_reported():
    with pytest.raises(EvaluationError, match="no 'test' set"):
        select_cases(MANIFEST, which="test", limit=None, seed=0, allow_test=True)
    with pytest.raises(EvaluationError, match="no 'full' set"):
        select_cases(V2, which="full", limit=None, seed=0, allow_full=True)


def test_a_test_set_that_no_longer_matches_its_digest_is_refused():
    from miresolver.benchmark import digest_cases

    manifest = json.loads(json.dumps(V2))
    manifest["meta"]["test_sha256"] = digest_cases(manifest["test"])
    assert select_cases(manifest, which="test", limit=None, seed=0, allow_test=True)
    manifest["test"][0]["changed_files"] = ["src/other.ts"]
    with pytest.raises(EvaluationError, match="digest"):
        select_cases(manifest, which="test", limit=None, seed=0, allow_test=True)


def test_retrieval_check_reports_precedent_availability_at_each_depth(tmp_path):
    from miresolver.evaluation.runner import run_retrieval_check

    cases = select_cases(MANIFEST, which="dev", limit=2, seed=1)
    first, second = cases

    async def retrieve(issue, exclude, k):
        assert exclude(cand(issue.number, 1))
        if issue.number == first.issue_number:
            files = [("x.ts",), ("y.ts",), (f"src/{first.issue_number}.ts",)]
        else:
            files = [("x.ts",)]
        return [Candidate(100 + i, "t", "b", 5, f, i) for i, f in enumerate(files)][:k]

    out = asyncio.run(run_retrieval_check(
        cases, issue_text=lambda n: ("t", "b"), retrieve=retrieve,
        out_root=tmp_path, run_meta={"set": "dev", "seed": 1},
        depths=(1, 3), timestamp="20261010T000000Z",
    ))
    assert out.name == "retrieval-dev-n2-seed1-20261010T000000Z"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["precedent_available_at"] == {"1": 0.0, "3": 0.5}
    assert summary["counts"]["scored"] == 2
    lines = [json.loads(l) for l in (out / "cases.jsonl").read_text().splitlines()]
    assert lines[0]["first_precedent_rank"] == 3
    assert lines[1]["first_precedent_rank"] is None


def test_a_run_tag_is_appended_to_the_directory_name(tmp_path):
    cases = select_cases(MANIFEST, which="dev", limit=1, seed=1)

    async def resolver(issue, exclude):
        return resolution([])

    out = asyncio.run(run_evaluation(
        cases, issue_text=lambda n: ("t", "b"), resolver=resolver, settings=SETTINGS,
        out_root=tmp_path, run_meta={"set": "dev", "seed": 3, "tag": "always-answer"},
        timestamp="20261010T000000Z",
    ))
    assert out.name == "dev-n1-seed3-20261010T000000Z-always-answer"


def test_case_lines_record_each_candidates_changed_files(tmp_path):
    cases = select_cases(MANIFEST, which="dev", limit=1, seed=1)

    async def resolver(issue, exclude):
        return resolution([], cands=[Candidate(7, "", "", 70, ("src/a.ts",), 0)])

    out = run(cases, resolver, tmp_path)
    line = json.loads((out / "cases.jsonl").read_text().splitlines()[0])
    assert line["candidates"] == [
        {"issue_number": 7, "pr_number": 70, "changed_files": ["src/a.ts"]}
    ]

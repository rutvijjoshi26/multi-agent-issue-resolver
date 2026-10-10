import asyncio

from agent_fakes import ChatError, ScriptedChat, Usage

from miresolver.agents.synthesize import merge_verdict_files, synthesize
from miresolver.agents.types import Citation, NewIssue, Verdict

ISSUE = NewIssue(title="Crash on save", body="trace", base_commit="b" * 40)


def verdict(n, *, applies=True, conf=0.8, files=("a.ts",), error=None, approach=None):
    return Verdict(n, n * 10, applies, conf, tuple(files),
                   approach or f"approach {n}", f"why {n}", error)


def run(verdicts, chat, threshold=0.5):
    return asyncio.run(synthesize(ISSUE, verdicts, chat, threshold=threshold))


def test_merge_ranks_files_by_summed_confidence_then_first_appearance():
    vs = [verdict(1, conf=0.4, files=("x.ts", "y.ts")),
          verdict(2, conf=0.5, files=("y.ts", "z.ts")),
          verdict(3, conf=0.5, files=("w.ts",))]
    assert merge_verdict_files(vs) == ["y.ts", "z.ts", "w.ts", "x.ts"]


def test_abstains_without_calling_the_model_when_no_precedent_is_confident():
    chat = ScriptedChat([])
    result = run([verdict(1, conf=0.4, files=("x.ts",)), verdict(2, applies=False, conf=0.9)], chat)
    assert chat.calls == []
    assert result.abstained and result.reason == "no_confident_precedent"
    assert result.files == ("x.ts", "a.ts")
    assert result.usage == Usage()


def test_abstains_when_every_investigator_failed():
    result = run([verdict(1, error="timeout"), verdict(2, error="boom")], ScriptedChat([]))
    assert result.abstained and result.reason == "no_verdicts"
    assert result.files == ()


def test_uses_the_models_ranking_and_filters_citations_to_investigated_issues():
    chat = ScriptedChat([{"approach": "guard null", "files": ["b.ts", "a.ts"],
                          "cited_issues": [2, 99, "1"], "confidence": 0.7}])
    result = run([verdict(1), verdict(2, files=("b.ts",))], chat)
    assert not result.abstained and result.reason is None
    assert result.files == ("b.ts", "a.ts")
    assert result.citations == (Citation(2, 20), Citation(1, 10))
    assert result.confidence == 0.7 and result.approach == "guard null"
    assert result.usage == Usage(10, 5)
    assert not result.fallback


def test_failed_verdicts_are_left_out_of_the_synthesis_prompt():
    chat = ScriptedChat([{"approach": "a", "files": ["a.ts"], "cited_issues": [1], "confidence": 0.9}])
    run([verdict(1), verdict(3, error="timed out")], chat)
    prompt = chat.calls[0]["messages"][0]["content"]
    assert "#1" in prompt and "#3" not in prompt


def test_low_synthesis_confidence_abstains_but_keeps_the_would_be_answer():
    chat = ScriptedChat([{"approach": "maybe", "files": ["a.ts"], "cited_issues": [1], "confidence": 0.3}])
    result = run([verdict(1)], chat)
    assert result.abstained and result.reason == "low_synthesis_confidence"
    assert result.files == ("a.ts",)


def test_empty_model_files_and_citations_fall_back_to_the_verdicts():
    chat = ScriptedChat([{"approach": "x", "files": [], "cited_issues": [], "confidence": 0.9}])
    result = run([verdict(1, files=("q.ts",)), verdict(2, conf=0.2, files=("r.ts",))], chat)
    assert result.files == ("q.ts", "r.ts")
    assert result.citations == (Citation(1, 10),)


def test_a_failed_synthesis_falls_back_to_merging_confident_verdicts():
    for reply in (ChatError("down"), "garbage", {"files": ["a.ts"], "confidence": "sure"}):
        result = run([verdict(1, conf=0.9, files=("p.ts",)), verdict(2, conf=0.6, files=("o.ts",)),
                      verdict(3, conf=0.3, files=("n.ts",))], ScriptedChat([reply]))
        assert result.fallback
        assert not result.abstained
        assert result.files == ("p.ts", "o.ts")
        assert result.confidence == 0.9 and result.approach == "approach 1"
        assert result.citations == (Citation(1, 10), Citation(2, 20))


def test_an_empty_file_list_is_not_rendered_as_a_path_like_word():
    chat = ScriptedChat([{"approach": "a", "files": ["a.ts"], "cited_issues": [1], "confidence": 0.9}])
    run([verdict(1), verdict(2, files=())], chat)
    prompt = chat.calls[0]["messages"][0]["content"]
    assert "Files: none" not in prompt
    assert "no files named" in prompt


def test_unexpected_synthesis_exceptions_still_fall_back():
    result = run([verdict(1, conf=0.9, files=("p.ts",))], ScriptedChat([AttributeError("odd payload")]))
    assert result.fallback and result.files == ("p.ts",)


def test_a_non_list_cited_issues_value_does_not_crash():
    chat = ScriptedChat([{"approach": "a", "files": ["a.ts"], "cited_issues": 5, "confidence": 0.9}])
    result = run([verdict(1)], chat)
    assert result.citations == (Citation(1, 10),)


def run_always(verdicts, chat, threshold=0.5):
    return asyncio.run(
        synthesize(ISSUE, verdicts, chat, threshold=threshold, always_answer=True)
    )


SYNTH = {"approach": "guard null", "files": ["m.ts"], "cited_issues": [1], "confidence": 0.2}


def test_always_answer_synthesizes_even_without_a_confident_precedent():
    chat = ScriptedChat([SYNTH])
    result = run_always([verdict(1, conf=0.3, files=("x.ts",))], chat)
    assert len(chat.calls) == 1
    assert not result.abstained
    assert result.files == ("m.ts",)
    # What the abstaining rule would have decided is kept for analysis.
    assert result.reason == "no_confident_precedent"


def test_always_answer_ignores_a_low_synthesis_confidence():
    chat = ScriptedChat([SYNTH])
    result = run_always([verdict(1, conf=0.9)], chat)
    assert not result.abstained and result.confidence == 0.2
    assert result.reason == "low_synthesis_confidence"


def test_always_answer_falls_back_to_the_verdicts_when_synthesis_fails():
    chat = ScriptedChat([ChatError("down")])
    result = run_always([verdict(1, applies=False, conf=0.3, files=("x.ts",))], chat)
    assert not result.abstained and result.fallback
    assert result.files == ("x.ts",)


def test_always_answer_with_no_usable_verdicts_answers_nothing():
    result = run_always([verdict(1, error="boom")], ScriptedChat([]))
    assert not result.abstained and result.files == ()
    assert result.reason == "no_verdicts"


def test_the_synthesis_gate_can_be_set_apart_from_the_verdict_gate():
    reply = dict(SYNTH, confidence=0.6)
    loose = asyncio.run(synthesize(ISSUE, [verdict(1, conf=0.9)], ScriptedChat([reply]),
                                   threshold=0.5))
    strict = asyncio.run(synthesize(ISSUE, [verdict(1, conf=0.9)], ScriptedChat([reply]),
                                    threshold=0.5, synthesis_threshold=0.7))
    assert not loose.abstained
    assert strict.abstained and strict.reason == "low_synthesis_confidence"


def test_agreement_counts_investigators_naming_a_top_three_file():
    from miresolver.agents.synthesize import agreement

    vs = [verdict(1, files=("a.ts", "b.ts")), verdict(2, files=("b.ts",)),
          verdict(3, files=("x.ts", "y.ts", "z.ts", "a.ts")), verdict(4, error="t")]
    assert agreement(("b.ts", "a.ts", "q.ts"), vs) == 2
    # Only an investigator's own first three files count.
    assert agreement(("z.ts",), vs) == 1
    assert agreement(("q.ts",), vs) == 0


def test_too_little_agreement_abstains():
    reply = {"approach": "a", "files": ["a.ts"], "cited_issues": [1], "confidence": 0.9}
    vs = [verdict(1, files=("a.ts",)), verdict(2, files=("b.ts",))]
    ok = asyncio.run(synthesize(ISSUE, vs, ScriptedChat([reply]), threshold=0.5,
                                min_agreement=1))
    low = asyncio.run(synthesize(ISSUE, vs, ScriptedChat([reply]), threshold=0.5,
                                 min_agreement=2))
    assert not ok.abstained and ok.agreement == 1
    assert low.abstained and low.reason == "insufficient_agreement"
    always = asyncio.run(synthesize(ISSUE, vs, ScriptedChat([reply]), threshold=0.5,
                                    min_agreement=2, always_answer=True))
    assert not always.abstained and always.reason == "insufficient_agreement"

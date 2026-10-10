import asyncio
import re
import time

from agent_fakes import FakeReader, ScriptedChat, Usage, hit

from miresolver.agents.graph import ResolverDeps, build_graph, resolve
from miresolver.agents.types import Citation, NewIssue, ResolverOptions

BASE = "c" * 40
ISSUE = NewIssue(title="Terminal hangs", body="on resize", base_commit=BASE)


def retriever_of(hits, seen=None):
    async def retrieve(query, limit):
        if seen is not None:
            seen.append((query, limit))
        return hits[:limit]
    return retrieve


def screen_all(prompt):
    numbers = [int(n) for n in re.findall(r"^#(\d+):", prompt, re.M)]
    return {"scores": [{"issue": n, "score": 1.0 - n / 100} for n in numbers]}


def verdict_for(prompt):
    number = int(re.search(r"PAST ISSUE #(\d+)", prompt).group(1))
    return {"applies": True, "confidence": 0.9, "files": [f"src/f{number}.ts"],
            "approach": f"as in #{number}", "reasoning": "match"}


SYNTH = {"approach": "fix resize", "files": ["src/f1.ts", "src/f2.ts"],
         "cited_issues": [1, 2], "confidence": 0.8}


def deps(hits, *, investigate=None, synth=None, options=None, seen=None):
    return ResolverDeps(
        retriever=retriever_of(hits, seen),
        screen_chat=ScriptedChat(respond=screen_all),
        investigate_chat=investigate or ScriptedChat(respond=verdict_for),
        synthesize_chat=synth or ScriptedChat([SYNTH]),
        reader=FakeReader(),
        options=options or ResolverOptions(),
    )


def test_end_to_end_produces_ranked_files_with_citations():
    d = deps([hit(1), hit(2), hit(3), hit(4)])
    result = asyncio.run(resolve(ISSUE, d))
    assert not result.abstained
    assert result.files == ("src/f1.ts", "src/f2.ts")
    assert result.citations == (Citation(1, 1001), Citation(2, 1002))
    assert [c.issue_number for c in result.candidates] == [1, 2, 3, 4]
    assert [c.issue_number for c in result.screened] == [1, 2, 3]
    assert [v.issue_number for v in result.verdicts] == [1, 2, 3]
    assert result.usage == {"screen": Usage(10, 5), "investigate": Usage(30, 15),
                            "synthesize": Usage(10, 5)}


def test_the_query_is_the_issue_text_and_k_comes_from_options():
    seen = []
    asyncio.run(resolve(ISSUE, deps([hit(1)], seen=seen, options=ResolverOptions(retrieve_k=4))))
    assert seen[0][0].startswith("Terminal hangs")
    assert "on resize" in seen[0][0]
    assert seen[0][1] >= 4


def test_investigators_run_concurrently():
    slow = ScriptedChat(respond=verdict_for, delay=0.3)
    d = deps([hit(1), hit(2), hit(3)], investigate=slow)
    started = time.perf_counter()
    asyncio.run(resolve(ISSUE, d))
    assert time.perf_counter() - started < 0.75
    assert len(slow.calls) == 3


def test_a_hung_investigator_times_out_and_the_rest_are_used():
    def respond(prompt):
        if "PAST ISSUE #2" in prompt:
            async def hang():
                await asyncio.sleep(30)
            return hang
        return verdict_for(prompt)

    d = deps([hit(1), hit(2), hit(3)], investigate=ScriptedChat(respond=respond),
             options=ResolverOptions(investigator_timeout_s=0.3))
    started = time.perf_counter()
    result = asyncio.run(resolve(ISSUE, d))
    assert time.perf_counter() - started < 2
    failed = [v for v in result.verdicts if v.failed]
    assert [v.issue_number for v in failed] == [2]
    assert "timed out" in failed[0].error
    assert not result.abstained


def test_no_candidates_abstains_without_any_model_call():
    d = deps([])
    result = asyncio.run(resolve(ISSUE, d))
    assert result.abstained and result.abstain_reason == "no_candidates"
    assert d.screen_chat.calls == [] and d.synthesize_chat.calls == []


def test_screening_everything_out_abstains():
    d = deps([hit(1), hit(2)])
    d.screen_chat.respond = lambda p: {"scores": []}
    result = asyncio.run(resolve(ISSUE, d))
    assert result.abstained and result.abstain_reason == "no_candidates_after_screening"
    assert d.investigate_chat.calls == []


def test_exclude_removes_candidates_before_screening_and_backfills_to_k():
    d = deps([hit(n) for n in range(1, 9)], options=ResolverOptions(retrieve_k=3))
    result = asyncio.run(resolve(ISSUE, d, exclude=lambda c: c.issue_number in (1, 3)))
    assert [c.issue_number for c in result.candidates] == [2, 4, 5]


def test_build_graph_compiles():
    assert build_graph(deps([])) is not None


def unsolved(number):
    h = hit(number, files=())
    payload = dict(h.payload, pr_number=None)
    return type(h)(issue_number=h.issue_number, score=h.score, payload=payload)


def test_issues_without_a_merged_fix_are_never_treated_as_precedents():
    hits = [unsolved(1), hit(2), unsolved(3), hit(4), hit(5), hit(6)]
    d = deps(hits, options=ResolverOptions(retrieve_k=3))
    result = asyncio.run(resolve(ISSUE, d))
    assert [c.issue_number for c in result.candidates] == [2, 4, 5]
    assert all(c.pr_number is not None for c in result.screened)


def test_timed_out_investigators_are_counted_as_unmetered():
    def respond(prompt):
        if "PAST ISSUE #2" in prompt:
            async def hang():
                await asyncio.sleep(30)
            return hang
        return verdict_for(prompt)

    d = deps([hit(1), hit(2)], investigate=ScriptedChat(respond=respond),
             options=ResolverOptions(investigator_timeout_s=0.2))
    result = asyncio.run(resolve(ISSUE, d))
    assert result.unmetered_calls == 1


def test_retrieval_widens_until_k_precedents_survive_the_filters():
    # Most corpus issues have no merged fix, and the leakage filter removes
    # more, so a fixed overfetch can return far fewer than k candidates.
    # Here only every 50th hit is a usable precedent.
    hits = [hit(n) if n % 50 == 0 else unsolved(n) for n in range(1, 1001)]
    seen = []
    d = deps(hits, options=ResolverOptions(retrieve_k=5), seen=seen)
    result = asyncio.run(resolve(ISSUE, d))
    assert [c.issue_number for c in result.candidates] == [50, 100, 150, 200, 250]
    assert [c.retrieval_rank for c in result.candidates] == [0, 1, 2, 3, 4]
    assert len(seen) > 1


def test_retrieval_stops_widening_once_the_index_is_exhausted():
    hits = [unsolved(n) for n in range(1, 31)] + [hit(31)]
    seen = []
    d = deps(hits, options=ResolverOptions(retrieve_k=5), seen=seen)
    result = asyncio.run(resolve(ISSUE, d))
    assert [c.issue_number for c in result.candidates] == [31]
    # 20 asked for, then 80; only 31 exist, so the second fetch was the last.
    assert [limit for _, limit in seen] == [20, 80]


def test_retrieval_does_not_widen_when_the_first_fetch_suffices():
    seen = []
    d = deps([hit(n) for n in range(1, 60)], options=ResolverOptions(retrieve_k=3), seen=seen)
    asyncio.run(resolve(ISSUE, d))
    assert len(seen) == 1


def test_always_answer_never_abstains_even_with_no_candidates():
    d = deps([], options=ResolverOptions(always_answer=True))
    result = asyncio.run(resolve(ISSUE, d))
    assert not result.abstained and result.files == ()
    assert result.abstain_reason == "no_candidates"


def test_always_answer_synthesizes_a_low_confidence_case():
    low = ScriptedChat(respond=lambda p: {**verdict_for(p), "confidence": 0.1})
    d = deps([hit(1), hit(2)], investigate=low,
             options=ResolverOptions(always_answer=True))
    result = asyncio.run(resolve(ISSUE, d))
    assert not result.abstained
    assert result.files == ("src/f1.ts", "src/f2.ts")
    assert len(d.synthesize_chat.calls) == 1


def test_the_synthesis_threshold_option_reaches_the_synthesizer():
    d = deps([hit(1), hit(2)], options=ResolverOptions(synthesis_threshold=0.9))
    result = asyncio.run(resolve(ISSUE, d))
    assert result.abstained and result.abstain_reason == "low_synthesis_confidence"


def test_retrieve_candidates_applies_the_same_filters_as_the_graph():
    from miresolver.agents.graph import retrieve_candidates

    hits = [hit(1), hit(2, files=()), hit(3), hit(4)]
    found = asyncio.run(retrieve_candidates(
        retriever_of(hits), ISSUE, k=2, exclude=lambda c: c.issue_number == 1,
    ))
    assert [c.issue_number for c in found] == [3, 4]
    assert [c.retrieval_rank for c in found] == [0, 1]

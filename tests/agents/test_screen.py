import asyncio

from agent_fakes import ChatError, ScriptedChat, Usage, hit

from miresolver.agents.screen import screen
from miresolver.agents.types import Candidate, NewIssue

ISSUE = NewIssue(title="Editor freezes", body="steps", base_commit="b" * 40)


def candidates(*numbers):
    return [Candidate.from_hit(hit(n), rank) for rank, n in enumerate(numbers)]


def test_candidate_from_hit_carries_the_fix_payload():
    c = Candidate.from_hit(hit(7, pr=70, files=("x.ts", "y.ts")), 3)
    assert (c.issue_number, c.pr_number, c.changed_files, c.retrieval_rank) == (
        7, 70, ("x.ts", "y.ts"), 3,
    )
    assert c.title == "issue 7"


def test_keeps_the_top_n_by_score_above_the_floor():
    chat = ScriptedChat([{"scores": [
        {"issue": 1, "score": 0.1}, {"issue": 2, "score": 0.9},
        {"issue": 3, "score": 0.5}, {"issue": 4, "score": 0.7},
    ]}])
    kept, usage = asyncio.run(
        screen(ISSUE, candidates(1, 2, 3, 4), chat, top_n=3, floor=0.2)
    )
    assert [c.issue_number for c in kept] == [2, 4, 3]
    assert [c.screen_score for c in kept] == [0.9, 0.7, 0.5]
    assert usage == Usage(10, 5)
    assert len(chat.calls) == 1 and chat.calls[0]["json_mode"]
    prompt = chat.calls[0]["messages"][0]["content"]
    assert "Editor freezes" in prompt and "#1" in prompt and "#4" in prompt


def test_omitted_candidates_score_zero_and_scores_are_clamped():
    chat = ScriptedChat([{"scores": [{"issue": 2, "score": 7}, {"issue": 99, "score": 1}]}])
    kept, _ = asyncio.run(screen(ISSUE, candidates(1, 2), chat, top_n=3, floor=0.0))
    assert [(c.issue_number, c.screen_score) for c in kept] == [(2, 1.0), (1, 0.0)]


def test_ties_break_by_retrieval_rank():
    chat = ScriptedChat([{"scores": [{"issue": 5, "score": 0.5}, {"issue": 3, "score": 0.5}]}])
    kept, _ = asyncio.run(screen(ISSUE, candidates(3, 5), chat, top_n=2, floor=0.0))
    assert [c.issue_number for c in kept] == [3, 5]


def test_a_failed_call_falls_back_to_retrieval_order():
    chat = ScriptedChat([ChatError("down")])
    kept, usage = asyncio.run(
        screen(ISSUE, candidates(4, 2, 9, 1), chat, top_n=3, floor=0.5)
    )
    assert [c.issue_number for c in kept] == [4, 2, 9]
    assert all(c.screen_score is None for c in kept)
    assert usage == Usage()


def test_an_unparseable_reply_falls_back_but_counts_its_tokens():
    chat = ScriptedChat(["no idea"])
    kept, usage = asyncio.run(screen(ISSUE, candidates(4, 2), chat, top_n=1, floor=0.5))
    assert [c.issue_number for c in kept] == [4]
    assert usage == Usage(10, 5)


def test_no_candidates_makes_no_call():
    chat = ScriptedChat([])
    assert asyncio.run(screen(ISSUE, [], chat, top_n=3, floor=0.2)) == ([], Usage())
    assert chat.calls == []


def test_any_screening_failure_falls_back_not_just_chat_errors():
    chat = ScriptedChat([AttributeError("odd payload")])
    kept, _ = asyncio.run(screen(ISSUE, candidates(4, 2), chat, top_n=1, floor=0.5))
    assert [c.issue_number for c in kept] == [4]


def test_non_finite_numbers_in_scores_are_skipped_not_fatal():
    chat = ScriptedChat(['{"scores": [{"issue": 1e999, "score": 1}, {"issue": 2, "score": 1e999}]}'])
    kept, _ = asyncio.run(screen(ISSUE, candidates(4, 2), chat, top_n=2, floor=0.5))
    assert [(c.issue_number, c.screen_score) for c in kept] == [(2, 1.0)]

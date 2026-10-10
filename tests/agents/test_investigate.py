import asyncio

from agent_fakes import ChatError, FakeReader, ScriptedChat, Usage, hit

from miresolver.agents.investigate import investigate
from miresolver.agents.types import Candidate, NewIssue, ResolverOptions

BASE = "b" * 40
ISSUE = NewIssue(title="Tabs lose focus", body="after reload", base_commit=BASE)
CANDIDATE = Candidate.from_hit(
    hit(11, pr=1100, files=("src/a.ts", "src/gone.ts", "src/broken.ts")), 0
)
OPTIONS = ResolverOptions(file_char_budget=20)

GOOD = {
    "applies": True, "confidence": 0.8, "files": ["src/a.ts", "src/new.ts"],
    "approach": "debounce the watcher", "reasoning": "same symptom",
}


def run(chat, reader=None, candidate=CANDIDATE, options=OPTIONS):
    reader = reader or FakeReader({(BASE, "src/a.ts"): "x" * 100})
    return asyncio.run(investigate(ISSUE, candidate, chat, reader, options)), reader


def test_reads_the_precedent_files_at_the_new_issues_base_commit():
    chat = ScriptedChat([GOOD])
    _, reader = run(chat, FakeReader({(BASE, "src/a.ts"): "code"}, errors=["src/broken.ts"]))
    assert reader.reads == [(BASE, p) for p in CANDIDATE.changed_files]
    prompt = chat.calls[0]["messages"][0]["content"]
    assert "Tabs lose focus" in prompt and "#11" in prompt and "1100" in prompt
    assert "src/gone.ts" in prompt and "does not exist" in prompt
    assert "could not be read" in prompt


def test_file_content_is_truncated_to_the_budget():
    chat = ScriptedChat([GOOD])
    run(chat)
    prompt = chat.calls[0]["messages"][0]["content"]
    assert "x" * 20 in prompt and "x" * 21 not in prompt
    assert "truncated" in prompt


def test_only_the_first_max_files_are_read():
    chat = ScriptedChat([GOOD])
    _, reader = run(chat, options=ResolverOptions(max_files_per_candidate=1))
    assert reader.reads == [(BASE, "src/a.ts")]


def test_parses_a_structured_verdict():
    (verdict, usage), _ = run(ScriptedChat([GOOD]))
    assert verdict.issue_number == 11 and verdict.pr_number == 1100
    assert verdict.applies is True and verdict.confidence == 0.8
    assert verdict.files == ("src/a.ts", "src/new.ts")
    assert verdict.approach == "debounce the watcher"
    assert not verdict.failed
    assert usage == Usage(10, 5)


def test_confidence_is_clamped_and_bad_files_dropped():
    reply = {**GOOD, "confidence": 3, "files": ["ok.ts", 5, "../x", "", " sp.ts "]}
    (verdict, _), _ = run(ScriptedChat([reply]))
    assert verdict.confidence == 1.0
    assert verdict.files == ("ok.ts", "sp.ts")


def test_a_malformed_reply_is_a_failed_verdict_not_an_exception():
    (verdict, usage), _ = run(ScriptedChat(["not json"]))
    assert verdict.failed and "JSON" in verdict.error
    assert verdict.confidence == 0.0 and not verdict.applies
    assert usage == Usage(10, 5)


def test_a_non_numeric_confidence_is_a_failed_verdict():
    (verdict, _), _ = run(ScriptedChat([{**GOOD, "confidence": "high"}]))
    assert verdict.failed


def test_a_model_error_is_a_failed_verdict():
    (verdict, usage), _ = run(ScriptedChat([ChatError("rate limited")]))
    assert verdict.failed and "rate limited" in verdict.error
    assert usage == Usage()

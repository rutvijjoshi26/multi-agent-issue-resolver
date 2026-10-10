import re

import pytest

from miresolver.retrieval.documents import build_document
from miresolver.retrieval.index import (
    IssueIndex,
    SearchHit,
    default_collection_name,
    fuse,
)
from miresolver.rules import CandidateIssue


def hit(number: int, score: float) -> SearchHit:
    return SearchHit(issue_number=number, score=score, payload={})


def test_fuse_ranks_an_item_found_by_both_above_one_found_by_either():
    dense = [hit(1, 0.9), hit(2, 0.8)]
    keyword = [hit(3, 40.0), hit(2, 30.0)]
    fused = fuse(dense, keyword, limit=3)
    assert fused[0].issue_number == 2


def test_fuse_is_insensitive_to_score_scale():
    dense = [hit(1, 0.9), hit(2, 0.89)]
    keyword = [hit(2, 9999.0), hit(1, 1.0)]
    fused = fuse(dense, keyword, limit=2)
    assert {h.issue_number for h in fused} == {1, 2}
    assert fused[0].issue_number == 1


def test_fuse_respects_the_limit():
    dense = [hit(i, 1.0) for i in range(10)]
    assert len(fuse(dense, [], limit=3)) == 3


def test_fuse_deduplicates_by_issue_number():
    fused = fuse([hit(1, 0.9)], [hit(1, 50.0)], limit=5)
    assert len(fused) == 1


def test_fuse_keeps_the_payload_from_whichever_side_had_it():
    dense = [SearchHit(issue_number=1, score=0.9, payload={"title": "t"})]
    fused = fuse(dense, [hit(1, 5.0)], limit=1)
    assert fused[0].payload == {"title": "t"}


def test_upsert_rejects_mismatched_counts(qdrant):
    index = IssueIndex(qdrant, collection="test_mismatch")
    index.create(vector_size=3)
    issue = CandidateIssue(
        number=1, title="t", body="b", labels=("bug",), linked_pr_numbers=()
    )
    with pytest.raises(ValueError, match="one vector per document"):
        index.upsert([build_document(issue)], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


def test_round_trip_dense_search_returns_the_nearest_document(qdrant):
    index = IssueIndex(qdrant, collection="test_dense")
    index.create(vector_size=3)
    issues = [
        CandidateIssue(
            number=n, title=f"issue {n}", body="b", labels=("bug",),
            linked_pr_numbers=(),
        )
        for n in (1, 2)
    ]
    docs = [build_document(i) for i in issues]
    index.upsert(docs, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert index.count() == 2

    hits = index.search_dense([0.99, 0.01, 0.0], limit=1)
    assert [h.issue_number for h in hits] == [1]
    assert hits[0].payload["title"] == "issue 1"


@pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect in the local Qdrant.*:UserWarning"
)
def test_keyword_search_finds_an_exact_symbol(qdrant):
    # The in-memory Qdrant client makes create_payload_index a no-op and
    # answers MatchText with an in-process scan instead of a real text
    # index (hence the suppressed warning below). This test therefore
    # proves the exact-symbol matching behavior works against local mode,
    # but it cannot catch a wrong index type/tokenizer or an index that
    # was never created, since a real server would still need the index
    # to serve MatchText at all. See test_keyword_search_against_a_live_qdrant
    # for the coverage that exercises the real index path.
    index = IssueIndex(qdrant, collection="test_keyword")
    index.create(vector_size=3)
    issues = [
        CandidateIssue(
            number=1, title="ErrInvalidConfigFormat on startup", body="b",
            labels=("bug",), linked_pr_numbers=(),
        ),
        CandidateIssue(
            number=2, title="window flickers on resize", body="b",
            labels=("bug",), linked_pr_numbers=(),
        ),
    ]
    docs = [build_document(i) for i in issues]
    index.upsert(docs, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    hits = index.search_keyword("ErrInvalidConfigFormat", limit=5)
    assert [h.issue_number for h in hits] == [1]


def test_create_is_idempotent(qdrant):
    index = IssueIndex(qdrant, collection="test_idempotent")
    index.create(vector_size=3)
    index.create(vector_size=3)
    assert index.count() == 0


def test_keyword_search_against_a_live_qdrant(live_qdrant):
    """Exercises the real text index, which the in-memory client cannot.

    Skips unless TEST_QDRANT_URL is set (mirrors the conn fixture's
    TEST_DATABASE_URL gate). In-memory mode makes create_payload_index a
    no-op and answers MatchText via an in-process scan regardless of
    whether a text index exists, so it cannot catch a missing or
    misconfigured index; a real server needs the index to serve MatchText
    at all. This test pins the same exact-symbol behavior against a real
    server so that gap is covered.
    """
    index = IssueIndex(live_qdrant, collection="test_live_keyword")
    index.create(vector_size=3)
    issues = [
        CandidateIssue(
            number=1, title="ErrInvalidConfigFormat on startup", body="b",
            labels=("bug",), linked_pr_numbers=(),
        ),
        CandidateIssue(
            number=2, title="window flickers on resize", body="b",
            labels=("bug",), linked_pr_numbers=(),
        ),
    ]
    docs = [build_document(i) for i in issues]
    index.upsert(docs, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    hits = index.search_keyword("ErrInvalidConfigFormat", limit=5)
    assert [h.issue_number for h in hits] == [1]


def test_default_collection_name_is_stable_for_the_same_model():
    assert default_collection_name("text-embed-v1") == default_collection_name(
        "text-embed-v1"
    )


def test_default_collection_name_differs_across_models():
    a = default_collection_name("text-embed-v1")
    b = default_collection_name("text-embed-v2")
    assert a != b


def test_default_collection_name_is_prefixed_and_safe_for_qdrant():
    name = default_collection_name("Org/Some Weird Model:v1.0!")
    assert name.startswith("issues__")
    # Qdrant collection names are safest as ascii letters, digits, "_", "-".
    assert re.fullmatch(r"[a-z0-9_-]+", name)


def test_default_collection_name_handles_an_empty_model_id():
    name = default_collection_name("")
    assert name.startswith("issues__")
    assert re.fullmatch(r"[a-z0-9_-]+", name)

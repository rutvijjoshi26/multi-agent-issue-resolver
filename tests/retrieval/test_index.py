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


def test_keyword_search_finds_an_exact_symbol(qdrant):
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
    """The same exact-symbol behavior against a real server.

    Skips unless TEST_QDRANT_URL is set. The in-memory client implements
    sparse vectors and the IDF modifier in process, so this pins the
    behavior against the real server's implementation as well.
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


def _issue(number: int, title: str, body: str = "b") -> CandidateIssue:
    return CandidateIssue(
        number=number, title=title, body=body, labels=("bug",), linked_pr_numbers=()
    )


def _three_issue_index(client, collection: str) -> IssueIndex:
    index = IssueIndex(client, collection=collection)
    index.create(vector_size=3)
    docs = [
        build_document(_issue(1, "terminal freezes when resizing the panel")),
        build_document(_issue(2, "editor minimap renders blank after theme change")),
        build_document(_issue(3, "git blame decorations flicker in the editor")),
    ]
    index.upsert(docs, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return index


def test_keyword_search_matches_a_long_query_that_shares_only_some_words(qdrant):
    # The query is a whole bug report, so most of its words appear in no
    # stored issue. A conjunctive full-text filter matched nothing here,
    # which made hybrid retrieval dense-only.
    index = _three_issue_index(qdrant, "test_keyword_partial")
    query = (
        "After upgrading, the integrated terminal freezes completely whenever "
        "I drag to resize it. Steps: open a shell, drag the sash, observe."
    )
    hits = index.search_keyword(query, limit=3)
    assert hits and hits[0].issue_number == 1


def test_keyword_search_ranks_by_overlap(qdrant):
    index = _three_issue_index(qdrant, "test_keyword_rank")
    hits = index.search_keyword("editor minimap blank", limit=3)
    assert [h.issue_number for h in hits][:2] == [2, 3]
    assert hits[0].score > hits[1].score


def test_keyword_search_returns_nothing_for_unrelated_words(qdrant):
    index = _three_issue_index(qdrant, "test_keyword_none")
    assert index.search_keyword("zzzunrelated qqqwords", limit=3) == []


def test_keyword_search_on_a_collection_without_bm25_asks_for_a_reindex(qdrant):
    from qdrant_client import models

    from miresolver.retrieval.index import KeywordIndexMissing

    qdrant.create_collection(
        "test_keyword_legacy",
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )
    index = IssueIndex(qdrant, collection="test_keyword_legacy")
    assert not index.has_keyword_vectors()
    with pytest.raises(KeywordIndexMissing, match="miresolver index"):
        index.search_keyword("anything", limit=3)


def test_create_replaces_a_collection_without_bm25(qdrant):
    from qdrant_client import models

    qdrant.create_collection(
        "test_keyword_upgrade",
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )
    index = IssueIndex(qdrant, collection="test_keyword_upgrade")
    index.create(vector_size=3)
    assert index.has_keyword_vectors()


def test_ranked_keyword_search_against_a_live_qdrant(live_qdrant):
    index = _three_issue_index(live_qdrant, "test_live_keyword_rank")
    hits = index.search_keyword(
        "the integrated terminal freezes whenever I resize it", limit=3
    )
    assert hits and hits[0].issue_number == 1

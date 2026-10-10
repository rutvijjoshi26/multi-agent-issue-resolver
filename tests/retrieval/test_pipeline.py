import asyncio

import pytest

from miresolver.retrieval.documents import build_document
from miresolver.retrieval.index import IssueIndex
from miresolver.retrieval.pipeline import build_index, search
from miresolver.rules import CandidateIssue


class FakeEmbedder:
    """Maps text to a deterministic 3-vector; records call batches."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def embed(self, texts):
        self.batches.append(list(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    async def aclose(self) -> None:
        return None


def issues(count: int) -> list[CandidateIssue]:
    return [
        CandidateIssue(
            number=n, title=f"issue {n}", body="b" * n, labels=("bug",),
            linked_pr_numbers=(),
        )
        for n in range(1, count + 1)
    ]


def test_build_index_indexes_every_document(qdrant):
    index = IssueIndex(qdrant, collection="test_pipeline_all")
    docs = [build_document(i) for i in issues(5)]
    indexed = asyncio.run(build_index(docs, FakeEmbedder(), index))
    assert indexed == 5
    assert index.count() == 5


def test_build_index_batches_requests(qdrant):
    index = IssueIndex(qdrant, collection="test_pipeline_batch")
    embedder = FakeEmbedder()
    docs = [build_document(i) for i in issues(5)]
    asyncio.run(build_index(docs, embedder, index, batch_size=2))
    assert [len(b) for b in embedder.batches] == [2, 2, 1]


def test_build_index_handles_an_empty_corpus(qdrant):
    index = IssueIndex(qdrant, collection="test_pipeline_empty")
    embedder = FakeEmbedder()
    assert asyncio.run(build_index([], embedder, index)) == 0
    assert embedder.batches == []


class FiveDimEmbedder:
    """Like FakeEmbedder, but returns 5-vectors, to prove size is inferred."""

    async def embed(self, texts):
        return [[0.0, 0.0, 0.0, 0.0, 1.0] for _ in texts]

    async def aclose(self) -> None:
        return None


def test_build_index_infers_vector_size_from_the_model(qdrant):
    index = IssueIndex(qdrant, collection="test_pipeline_size")
    docs = [build_document(i) for i in issues(1)]
    asyncio.run(build_index(docs, FiveDimEmbedder(), index))
    assert index.count() == 1
    info = qdrant.get_collection("test_pipeline_size")
    assert info.config.params.vectors.size == 5


def test_search_returns_fused_hits(qdrant):
    index = IssueIndex(qdrant, collection="test_pipeline_search")
    docs = [build_document(i) for i in issues(3)]
    embedder = FakeEmbedder()
    asyncio.run(build_index(docs, embedder, index))
    hits = asyncio.run(search("issue 2", embedder, index, limit=3))
    assert hits
    assert all(h.issue_number in {1, 2, 3} for h in hits)


def test_search_on_an_empty_index_returns_nothing(qdrant):
    index = IssueIndex(qdrant, collection="test_pipeline_searchempty")
    index.create(vector_size=3)
    assert asyncio.run(search("anything", FakeEmbedder(), index)) == []


def test_search_after_indexing_an_empty_corpus_returns_nothing(qdrant):
    """The collection is never created when the corpus is empty; search
    must not surface that as a raw "collection not found" error."""
    index = IssueIndex(qdrant, collection="test_pipeline_empty_then_search")
    embedder = FakeEmbedder()
    assert asyncio.run(build_index([], embedder, index)) == 0
    assert not index.exists()
    assert asyncio.run(search("anything", embedder, index)) == []


class RecordingIndex:
    def __init__(self, inner):
        self.inner = inner
        self.avgdls = []

    def create(self, vector_size):
        self.inner.create(vector_size)

    def upsert(self, documents, vectors, *, avgdl=None):
        self.avgdls.append(avgdl)
        self.inner.upsert(documents, vectors, avgdl=avgdl)


def test_build_index_normalises_bm25_by_the_whole_corpus(qdrant):
    from miresolver.retrieval.keyword import average_length

    docs = [build_document(i) for i in issues(4)]
    recording = RecordingIndex(IssueIndex(qdrant, collection="test_pipeline_avgdl"))
    asyncio.run(build_index(docs, FakeEmbedder(), recording, batch_size=1))
    expected = average_length([d.embed_text for d in docs])
    assert recording.avgdls == [expected] * 4


def test_search_fuses_keyword_hits_the_dense_side_missed(qdrant):
    index = IssueIndex(qdrant, collection="test_pipeline_hybrid")
    corpus = [
        CandidateIssue(
            number=n, title=title, body="", labels=("bug",), linked_pr_numbers=()
        )
        for n, title in (
            (1, "ErrInvalidConfigFormat thrown at startup"),
            (2, "window flickers"),
            (3, "menu missing"),
        )
    ]
    embedder = FakeEmbedder()
    asyncio.run(build_index([build_document(i) for i in corpus], embedder, index))
    hits = asyncio.run(search("ErrInvalidConfigFormat", embedder, index, limit=1))
    # The fake embedder's vectors carry no meaning, so only the keyword
    # side can put issue 1 first.
    assert [h.issue_number for h in hits] == [1]


def _hybrid_corpus(qdrant, collection):
    index = IssueIndex(qdrant, collection=collection)
    corpus = [
        CandidateIssue(n, title, "", ("bug",), ())
        for n, title in ((1, "ErrInvalidConfigFormat thrown"), (2, "window flickers"))
    ]
    embedder = FakeEmbedder()
    asyncio.run(build_index([build_document(i) for i in corpus], embedder, index))
    return index, embedder


def test_search_can_run_dense_only_or_keyword_only(qdrant):
    index, embedder = _hybrid_corpus(qdrant, "test_pipeline_modes")
    keyword = asyncio.run(
        search("ErrInvalidConfigFormat", embedder, index, limit=5, mode="keyword")
    )
    assert [h.issue_number for h in keyword] == [1]
    dense = asyncio.run(search("ErrInvalidConfigFormat", embedder, index, limit=5, mode="dense"))
    assert {h.issue_number for h in dense} == {1, 2}


def test_search_rejects_an_unknown_mode(qdrant):
    index, embedder = _hybrid_corpus(qdrant, "test_pipeline_badmode")
    with pytest.raises(ValueError, match="mode"):
        asyncio.run(search("x", embedder, index, limit=5, mode="fuzzy"))

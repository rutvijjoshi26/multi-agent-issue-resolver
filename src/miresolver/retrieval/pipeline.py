from collections.abc import Sequence

from miresolver.models.embedding import EmbeddingClient
from miresolver.retrieval.documents import IssueDocument
from miresolver.retrieval.index import IssueIndex, SearchHit, fuse
from miresolver.retrieval.keyword import average_length


async def build_index(
    documents: Sequence[IssueDocument],
    embedder: EmbeddingClient,
    index: IssueIndex,
    *,
    batch_size: int = 64,
    vector_size: int | None = None,
) -> int:
    """Embed every document and write it to the index.

    The vector size comes from the first batch rather than a constant,
    because it is a property of whichever model the user configured.
    """
    if not documents:
        return 0

    # BM25 normalises each document by the corpus-wide average length, so
    # it is computed once here rather than per batch.
    avgdl = average_length([doc.embed_text for doc in documents])
    indexed = 0
    created = False
    for start in range(0, len(documents), batch_size):
        batch = list(documents[start : start + batch_size])
        vectors = await embedder.embed([doc.embed_text for doc in batch])
        if not created:
            size = vector_size if vector_size is not None else len(vectors[0])
            index.create(vector_size=size)
            created = True
        index.upsert(batch, vectors, avgdl=avgdl)
        indexed += len(batch)
    return indexed


SEARCH_MODES = ("hybrid", "dense", "keyword")


async def search(
    query: str,
    embedder: EmbeddingClient,
    index: IssueIndex,
    *,
    limit: int = 10,
    mode: str = "hybrid",
) -> list[SearchHit]:
    """Dense and keyword search over the same query, fused by rank.

    mode "dense" or "keyword" runs one side alone, for ablations.

    A never-created collection (e.g. an empty corpus was indexed and nothing
    was ever written) is treated as an empty index rather than an error: the
    caller gets no hits back instead of a raw "collection not found" failure.
    """
    if mode not in SEARCH_MODES:
        raise ValueError(f"unknown search mode {mode!r}; use one of {SEARCH_MODES}")
    if not index.exists():
        return []
    dense: list[SearchHit] = []
    keyword: list[SearchHit] = []
    if mode in ("hybrid", "dense"):
        vectors = await embedder.embed([query])
        dense = index.search_dense(vectors[0], limit=limit)
    if mode in ("hybrid", "keyword"):
        keyword = index.search_keyword(query, limit=limit)
    if mode == "dense":
        return dense
    if mode == "keyword":
        return keyword
    return fuse(dense, keyword, limit=limit)

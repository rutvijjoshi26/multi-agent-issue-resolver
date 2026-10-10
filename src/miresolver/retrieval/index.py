import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from miresolver.retrieval.documents import IssueDocument

DEFAULT_COLLECTION = "issues"

_UNSAFE_CHARS = re.compile(r"[^a-zA-Z0-9_-]+")


def default_collection_name(embed_model: str) -> str:
    """Derive a per-model collection name so switching embedding models
    cannot silently mix vectors from two different models in one collection.

    A user who points the embed tier at a different model (bring-your-own
    is the whole point of this project) is switching to a different vector
    space. Reusing a bare "issues" collection across that switch would let
    Qdrant accept the new vectors, with no error and no signal, right next
    to vectors from the old model, and later searches would silently score
    against both. Naming the collection after the model instead makes the
    two spaces structurally separate: the new model gets its own empty
    collection, and the old one is left for the user to delete once they
    are done with it.
    """
    sanitized = _UNSAFE_CHARS.sub("-", embed_model).strip("-").lower()
    if not sanitized:
        sanitized = "model"
    return f"issues__{sanitized}"


@dataclass(frozen=True)
class SearchHit:
    issue_number: int
    score: float
    payload: Mapping[str, object]


def fuse(
    dense: Sequence[SearchHit],
    keyword: Sequence[SearchHit],
    *,
    limit: int,
    k: int = 60,
) -> list[SearchHit]:
    """Reciprocal rank fusion.

    Dense cosine scores and keyword scores are not on a comparable scale, so
    ranks are combined rather than the scores themselves.
    """
    scores: dict[int, float] = {}
    payloads: dict[int, Mapping[str, object]] = {}
    for ranking in (dense, keyword):
        for rank, hit in enumerate(ranking):
            scores[hit.issue_number] = scores.get(hit.issue_number, 0.0) + 1.0 / (
                k + rank + 1
            )
            if hit.payload and hit.issue_number not in payloads:
                payloads[hit.issue_number] = hit.payload
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [
        SearchHit(
            issue_number=number, score=score, payload=payloads.get(number, {})
        )
        for number, score in ordered[:limit]
    ]


class IssueIndex:
    def __init__(
        self, client: QdrantClient, *, collection: str = DEFAULT_COLLECTION
    ) -> None:
        self._client = client
        self._collection = collection

    def exists(self) -> bool:
        return self._client.collection_exists(self._collection)

    def create(self, vector_size: int) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="embed_text",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
            ),
        )

    def upsert(
        self,
        documents: Sequence[IssueDocument],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(documents) != len(vectors):
            raise ValueError(
                f"one vector per document required: {len(documents)} documents, "
                f"{len(vectors)} vectors"
            )
        points = [
            models.PointStruct(
                id=doc.issue_number,
                vector=list(vector),
                payload={**doc.payload, "embed_text": doc.embed_text},
            )
            for doc, vector in zip(documents, vectors, strict=True)
        ]
        if points:
            self._client.upsert(collection_name=self._collection, points=points)

    def _to_hits(self, points: Sequence[object]) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            payload.pop("embed_text", None)
            hits.append(
                SearchHit(
                    issue_number=int(payload.get("issue_number", getattr(point, "id"))),
                    score=float(getattr(point, "score", 0.0)),
                    payload=payload,
                )
            )
        return hits

    def search_dense(
        self, vector: Sequence[float], *, limit: int
    ) -> list[SearchHit]:
        result = self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            limit=limit,
            with_payload=True,
        )
        return self._to_hits(result.points)

    def search_keyword(self, text: str, *, limit: int) -> list[SearchHit]:
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="embed_text", match=models.MatchText(text=text)
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        )
        return self._to_hits(points)

    def count(self) -> int:
        return self._client.count(
            collection_name=self._collection, exact=True
        ).count

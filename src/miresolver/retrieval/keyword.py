"""BM25 keyword matching expressed as Qdrant sparse vectors.

Each document is stored as a sparse vector of saturated, length-normalised
term frequencies; Qdrant's IDF modifier supplies the inverse document
frequency at query time, so a dot product with a query vector of ones is
the BM25 score. Unlike a full-text filter, this ranks documents and does
not require every query word to be present, which matters because the
query is a whole bug report.
"""

import re
import zlib
from collections import Counter

K1 = 1.2
B = 0.75

_WORD = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")

STOPWORDS = frozenset(
    """
    a about after again all also am an and any are as at be because been
    before being but by can could did do does doing done for from had has
    have having he her here him his how i if in into is it its just me
    more most my no nor not of off on once only or other our out over own
    same she should so some such than that the their them then there these
    they this those through to too under until up very was we were what
    when where which while who why will with would you your
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercased terms, keeping whole identifiers and their parts.

    Bug reports are full of symbols such as ErrInvalidConfigFormat or
    file_watcher.ts. The whole identifier is kept so an exact symbol match
    scores strongly, and its camelCase and snake_case parts are added so a
    report that only says "config format" can still match.
    """
    tokens: list[str] = []
    for word in _WORD.findall(text):
        lowered = word.lower()
        parts = [p.lower() for piece in word.split("_") for p in _CAMEL.findall(piece)]
        for term in dict.fromkeys([lowered, *parts]):
            if len(term) < 2 or term in STOPWORDS:
                continue
            tokens.append(term)
    return tokens


def term_index(term: str) -> int:
    """Stable 31-bit index for a term; collisions are rare and harmless."""
    return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF


def bm25_document_vector(
    text: str, *, avgdl: float, k1: float = K1, b: float = B
) -> tuple[list[int], list[float]]:
    counts = Counter(term_index(t) for t in tokenize(text))
    if not counts:
        return [], []
    length = sum(counts.values())
    norm = k1 * (1 - b + b * length / avgdl) if avgdl > 0 else k1
    indices = sorted(counts)
    values = [counts[i] * (k1 + 1) / (counts[i] + norm) for i in indices]
    return indices, values


def query_vector(text: str) -> tuple[list[int], list[float]]:
    indices = sorted({term_index(t) for t in tokenize(text)})
    return indices, [1.0] * len(indices)


def average_length(texts: list[str]) -> float:
    lengths = [len(tokenize(t)) for t in texts]
    return sum(lengths) / len(lengths) if lengths else 0.0

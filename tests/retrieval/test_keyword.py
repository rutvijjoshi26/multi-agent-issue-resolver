import math

from miresolver.retrieval.keyword import (
    bm25_document_vector,
    query_vector,
    term_index,
    tokenize,
)


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Terminal is frozen") == ["terminal", "frozen"]


def test_tokenize_keeps_whole_identifiers_and_their_parts():
    tokens = tokenize("ErrInvalidConfigFormat in file_watcher.ts")
    for expected in (
        "errinvalidconfigformat", "err", "invalid", "config", "format",
        "file_watcher", "file", "watcher", "ts",
    ):
        assert expected in tokens
    assert "in" not in tokens


def test_tokenize_splits_paths_and_dotted_names():
    tokens = tokenize("at src/vs/editor/browser/view.ts:42 editor.action.formatDocument")
    assert {"src", "vs", "editor", "browser", "view", "action"} <= set(tokens)
    assert "formatdocument" in tokens and "document" in tokens


def test_term_index_is_stable_and_non_negative():
    assert term_index("terminal") == term_index("terminal")
    assert term_index("terminal") != term_index("editor")
    assert 0 <= term_index("terminal") < 2**31


def test_document_vector_saturates_repeated_terms():
    indices, values = bm25_document_vector("crash crash crash crash hang", avgdl=5.0)
    weights = dict(zip(indices, values))
    crash, hang = weights[term_index("crash")], weights[term_index("hang")]
    assert crash > hang
    # BM25 term frequency saturates: four occurrences weigh far less than
    # four times one occurrence.
    assert crash < 2 * hang


def test_document_vector_penalises_long_documents():
    short_i, short_v = bm25_document_vector("crash", avgdl=10.0)
    long_i, long_v = bm25_document_vector("crash " + "word " * 40, avgdl=10.0)
    short = dict(zip(short_i, short_v))[term_index("crash")]
    long = dict(zip(long_i, long_v))[term_index("crash")]
    assert long < short


def test_document_vector_of_empty_text_is_empty():
    assert bm25_document_vector("", avgdl=3.0) == ([], [])


def test_query_vector_weights_each_distinct_term_once():
    indices, values = query_vector("crash crash hang")
    assert sorted(indices) == sorted({term_index("crash"), term_index("hang")})
    assert all(math.isclose(v, 1.0) for v in values)

from miresolver.retrieval.documents import MAX_EMBED_CHARS, build_document
from miresolver.rules import CandidateIssue

ISSUE = CandidateIssue(
    number=4821,
    title="crash when config file is empty",
    body="Opening an empty settings file crashes the window.",
    labels=("bug",),
    linked_pr_numbers=(4830,),
)


def test_embed_text_contains_title_and_body():
    doc = build_document(ISSUE)
    assert "crash when config file is empty" in doc.embed_text
    assert "Opening an empty settings file" in doc.embed_text


def test_embed_text_excludes_the_resolution():
    doc = build_document(
        ISSUE,
        pr_number=4830,
        base_commit="abc123",
        changed_files=("src/config/parser.ts",),
    )
    assert "parser.ts" not in doc.embed_text
    assert "abc123" not in doc.embed_text
    assert "4830" not in doc.embed_text


def test_resolution_travels_in_the_payload():
    doc = build_document(
        ISSUE,
        pr_number=4830,
        base_commit="abc123",
        changed_files=("src/config/parser.ts",),
    )
    assert doc.payload["pr_number"] == 4830
    assert doc.payload["base_commit"] == "abc123"
    assert doc.payload["changed_files"] == ("src/config/parser.ts",)


def test_payload_carries_the_symptom_text_for_display():
    doc = build_document(ISSUE)
    assert doc.payload["title"] == "crash when config file is empty"
    assert doc.payload["issue_number"] == 4821


def test_embed_text_is_truncated_at_the_limit():
    long_issue = CandidateIssue(
        number=1,
        title="t",
        body="x" * (MAX_EMBED_CHARS * 2),
        labels=("bug",),
        linked_pr_numbers=(),
    )
    doc = build_document(long_issue)
    assert len(doc.embed_text) == MAX_EMBED_CHARS


def test_document_number_matches_the_issue():
    assert build_document(ISSUE).issue_number == 4821


def test_payload_outer_mapping_cannot_be_mutated():
    doc = build_document(ISSUE)
    try:
        doc.payload["new_key"] = "value"
        assert False, "Expected TypeError when assigning to payload"
    except TypeError:
        pass  # Expected: read-only mapping


def test_payload_outer_mapping_keys_cannot_be_deleted():
    doc = build_document(ISSUE)
    try:
        del doc.payload["pr_number"]
        assert False, "Expected TypeError when deleting from payload"
    except TypeError:
        pass  # Expected: read-only mapping


def test_payload_nested_collections_are_immutable():
    doc = build_document(
        ISSUE,
        changed_files=("src/config/parser.ts",),
    )
    try:
        doc.payload["changed_files"].append("src/other.ts")
        assert False, "Expected TypeError when mutating changed_files"
    except (TypeError, AttributeError):
        pass  # Expected: tuple has no append method or is read-only


def test_payload_nested_labels_are_immutable():
    doc = build_document(ISSUE)
    try:
        doc.payload["labels"].append("feature")
        assert False, "Expected TypeError when mutating labels"
    except (TypeError, AttributeError):
        pass  # Expected: tuple has no append method or is read-only

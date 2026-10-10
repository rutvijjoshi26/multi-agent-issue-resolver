from miresolver.retrieval.loader import load_documents
from miresolver.rules import CandidateIssue, CandidatePR
from miresolver.store import create_schema, save_issues

ISSUE = CandidateIssue(
    number=4821,
    title="crash when config file is empty",
    body="it crashes",
    labels=("bug",),
    linked_pr_numbers=(4830,),
)
PR = CandidatePR(
    number=4830,
    merged=True,
    title="fix parser crash",
    changed_files=("src/config/parser.ts",),
    base_commit="abc123",
)


def test_load_documents_joins_the_resolution_into_the_payload(conn):
    create_schema(conn)
    save_issues(conn, [ISSUE], {4830: PR})
    docs = load_documents(conn)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.issue_number == 4821
    assert doc.payload["pr_number"] == 4830
    assert doc.payload["base_commit"] == "abc123"
    # payload tuples are immutable, not lists.
    assert doc.payload["changed_files"] == ("src/config/parser.ts",)


def test_load_documents_does_not_embed_the_resolution(conn):
    create_schema(conn)
    save_issues(conn, [ISSUE], {4830: PR})
    doc = load_documents(conn)[0]
    assert "parser.ts" not in doc.embed_text
    assert "abc123" not in doc.embed_text


def test_load_documents_returns_empty_for_an_empty_corpus(conn):
    create_schema(conn)
    assert load_documents(conn) == []


def test_load_documents_handles_an_issue_with_no_merged_pr(conn):
    open_issue = CandidateIssue(
        number=1,
        title="still open",
        body="no fix yet",
        labels=(),
        linked_pr_numbers=(),
    )
    create_schema(conn)
    save_issues(conn, [open_issue], {})
    docs = load_documents(conn)
    assert len(docs) == 1
    assert docs[0].payload["pr_number"] is None
    assert docs[0].payload["changed_files"] == ()


def test_load_documents_does_not_duplicate_an_issue_with_multiple_linked_prs(conn):
    issue = CandidateIssue(
        number=99,
        title="two prs touched this",
        body="body",
        labels=(),
        linked_pr_numbers=(101, 102),
    )
    pr_unmerged = CandidatePR(
        number=101, merged=False, title="wip", changed_files=("a.py",),
    )
    pr_merged = CandidatePR(
        number=102,
        merged=True,
        title="the actual fix",
        changed_files=("b.py",),
        base_commit="deadbeef",
    )
    create_schema(conn)
    save_issues(conn, [issue], {101: pr_unmerged, 102: pr_merged})
    docs = load_documents(conn)
    assert len(docs) == 1
    assert docs[0].payload["pr_number"] == 102

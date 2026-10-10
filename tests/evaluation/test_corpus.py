from miresolver.evaluation.corpus import load_issue_text
from miresolver.rules import CandidateIssue
from miresolver.store import create_schema, save_issues


def test_loads_title_and_body_for_requested_issues_only(conn):
    create_schema(conn)
    issues = [CandidateIssue(n, f"title {n}", f"body {n}", ("bug",), ()) for n in (1, 2, 3)]
    save_issues(conn, issues, {})
    assert load_issue_text(conn, [3, 1, 99]) == {1: ("title 1", "body 1"), 3: ("title 3", "body 3")}


def test_no_numbers_means_no_query(conn):
    assert load_issue_text(conn, []) == {}

from dataclasses import replace

from miresolver.rules import CandidateIssue, CandidatePR
from miresolver.store import create_schema, load_issue_numbers, save_issues

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
    base_commit="9f1c2ab7d3e4f50617283940abcdef1234567890",
)


def test_save_and_load_round_trip(conn):
    create_schema(conn)
    save_issues(conn, [ISSUE], {4830: PR})
    assert load_issue_numbers(conn) == [4821]


def test_save_is_idempotent(conn):
    create_schema(conn)
    save_issues(conn, [ISSUE], {4830: PR})
    save_issues(conn, [ISSUE], {4830: PR})
    assert load_issue_numbers(conn) == [4821]


def _read_issue(conn, number: int):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT title, body, labels FROM issues WHERE number = %s", (number,)
        )
        return cur.fetchone()


def _read_pr(conn, number: int):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT merged, title, changed_files, base_commit "
            "FROM pull_requests WHERE number = %s",
            (number,),
        )
        return cur.fetchone()


def test_save_round_trips_every_column(conn):
    create_schema(conn)
    save_issues(conn, [ISSUE], {4830: PR})

    assert _read_issue(conn, 4821) == (ISSUE.title, ISSUE.body, list(ISSUE.labels))
    assert _read_pr(conn, 4830) == (
        PR.merged,
        PR.title,
        list(PR.changed_files),
        PR.base_commit,
    )


def test_second_save_leaves_the_values_intact(conn):
    """Key presence alone would still pass if the upsert clobbered the
    payload, so assert the values survive."""
    create_schema(conn)
    save_issues(conn, [ISSUE], {4830: PR})
    save_issues(conn, [ISSUE], {4830: PR})

    assert load_issue_numbers(conn) == [4821]
    assert _read_issue(conn, 4821) == (ISSUE.title, ISSUE.body, list(ISSUE.labels))
    assert _read_pr(conn, 4830) == (
        PR.merged,
        PR.title,
        list(PR.changed_files),
        PR.base_commit,
    )


def test_save_updates_changed_values(conn):
    create_schema(conn)
    save_issues(conn, [ISSUE], {4830: PR})
    updated = replace(PR, title="fix parser crash (take two)", base_commit="b" * 40)
    save_issues(conn, [replace(ISSUE, title="retitled")], {4830: updated})

    assert _read_issue(conn, 4821)[0] == "retitled"
    assert _read_pr(conn, 4830)[1:] == (
        updated.title,
        list(updated.changed_files),
        updated.base_commit,
    )


def test_issue_numbers_come_back_ascending(conn):
    """Two rows inserted high-first, so the assertion can tell ORDER BY
    from insertion order."""
    create_schema(conn)
    later = replace(ISSUE, number=9999, linked_pr_numbers=())
    earlier = replace(ISSUE, number=1, linked_pr_numbers=())
    save_issues(conn, [later, earlier], {4830: PR})

    assert load_issue_numbers(conn) == [1, 9999]

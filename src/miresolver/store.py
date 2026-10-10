from collections.abc import Mapping, Sequence
from importlib import resources

import psycopg

from miresolver.rules import CandidateIssue, CandidatePR


def create_schema(conn: psycopg.Connection) -> None:
    sql = resources.files("miresolver").joinpath("schema.sql").read_text()
    with conn.cursor() as cur:
        cur.execute(sql)


def save_issues(
    conn: psycopg.Connection,
    issues: Sequence[CandidateIssue],
    prs: Mapping[int, CandidatePR],
) -> None:
    with conn.cursor() as cur:
        for pr in prs.values():
            cur.execute(
                """
                INSERT INTO pull_requests
                    (number, merged, title, changed_files, base_commit)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (number) DO UPDATE SET
                    merged = EXCLUDED.merged,
                    title = EXCLUDED.title,
                    changed_files = EXCLUDED.changed_files,
                    base_commit = EXCLUDED.base_commit
                """,
                (
                    pr.number,
                    pr.merged,
                    pr.title,
                    list(pr.changed_files),
                    pr.base_commit,
                ),
            )
        for issue in issues:
            cur.execute(
                """
                INSERT INTO issues (number, title, body, labels)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (number) DO UPDATE SET
                    title = EXCLUDED.title,
                    body = EXCLUDED.body,
                    labels = EXCLUDED.labels
                """,
                (issue.number, issue.title, issue.body, list(issue.labels)),
            )
            for pr_number in issue.linked_pr_numbers:
                if pr_number not in prs:
                    continue
                cur.execute(
                    """
                    INSERT INTO issue_pr_links (issue_number, pr_number)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (issue.number, pr_number),
                )


def load_issue_numbers(conn: psycopg.Connection) -> list[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT number FROM issues ORDER BY number")
        return [row[0] for row in cur.fetchall()]

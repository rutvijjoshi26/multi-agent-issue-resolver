import psycopg

from miresolver.retrieval.documents import IssueDocument, build_document
from miresolver.rules import CandidateIssue

# DISTINCT ON (i.number) collapses to one row per issue even when an issue
# has links to more than one pull request (several linked PRs, or one linked
# PR that later turns out unmerged alongside a merged one): ordering by
# p.number NULLS LAST prefers a real merged PR over a null join result, and
# breaks ties on the PR's own number, so the choice is deterministic rather
# than depending on row order returned by Postgres.
QUERY = """
SELECT DISTINCT ON (i.number)
       i.number, i.title, i.body, i.labels,
       p.number, p.base_commit, p.changed_files
FROM issues i
LEFT JOIN issue_pr_links l ON l.issue_number = i.number
LEFT JOIN pull_requests p ON p.number = l.pr_number AND p.merged
ORDER BY i.number, p.number NULLS LAST
"""


def load_documents(conn: psycopg.Connection) -> list[IssueDocument]:
    """Read the corpus and join each issue to its single merged fixing PR.

    An issue with no merged PR (still open, or fixed by a PR GitHub never
    reported as merged) gets a null pr_number in its payload rather than
    being dropped: retrieval still wants such an issue as a symptom match,
    it just has no resolution to carry along.
    """
    with conn.cursor() as cur:
        cur.execute(QUERY)
        rows = cur.fetchall()

    documents: list[IssueDocument] = []
    for number, title, body, labels, pr_number, base_commit, changed in rows:
        issue = CandidateIssue(
            number=number,
            title=title,
            body=body,
            labels=tuple(labels or ()),
            linked_pr_numbers=(pr_number,) if pr_number else (),
        )
        documents.append(
            build_document(
                issue,
                pr_number=pr_number,
                base_commit=base_commit or "",
                changed_files=tuple(changed or ()),
            )
        )
    return documents

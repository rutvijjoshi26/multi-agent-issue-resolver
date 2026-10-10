import asyncio
from collections.abc import Sequence
from typing import Any

import httpx

from miresolver.rules import CandidateIssue, CandidatePR

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

#: HTTP statuses worth another attempt. 429 and the 5xx family are transient
#: on GitHub; 4xx otherwise means the request itself is wrong.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

#: GraphQL error types GitHub raises for load rather than for a bad query.
RETRYABLE_ERROR_TYPES = frozenset(
    {"RATE_LIMITED", "SERVICE_UNAVAILABLE", "INTERNAL", "TIMEOUT"}
)


class GitHubError(RuntimeError):
    """A GraphQL response carried errors, or retries ran out.

    GitHub answers a rate limit or a field timeout with HTTP 200 and an
    ``errors`` block, so this has to be checked explicitly. Any errors block
    is fatal even when partial data came back: a page whose linked-PR field
    timed out looks like a page of issues with no fixing PR, which would be
    silently dropped by the inclusion rules and bias the corpus while the
    run reported success.
    """

    def __init__(self, errors: Sequence[Any]) -> None:
        self.errors = list(errors)
        summary = "; ".join(
            str(e.get("message", e)) if isinstance(e, dict) else str(e)
            for e in self.errors
        )
        super().__init__(f"GitHub GraphQL error: {summary}")


def _is_retryable(errors: Sequence[Any]) -> bool:
    return any(
        isinstance(e, dict) and e.get("type") in RETRYABLE_ERROR_TYPES
        for e in errors
    )


LINKED_PR_PAGE_SIZE = 10
CHANGED_FILE_PAGE_SIZE = 100

# closedByPullRequestsReferences is GitHub's own "this pull request closes
# this issue" relation: closing keywords in the PR body plus PRs linked by
# hand in the UI. A CrossReferencedEvent, by contrast, fires when an issue
# number is merely mentioned, so a changelog PR listing fifty issues would
# become the "fix" for all fifty.
#
# includeClosedPrs keeps PRs that were closed without merging in the list,
# which is what lets the "exactly one merged PR" rule see them and count
# them as not-merged rather than never seeing them at all.
ISSUES_QUERY = """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    issues(states: CLOSED, first: 50, after: $after,
           orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        body
        labels(first: 20) { nodes { name } }
        closedByPullRequestsReferences(first: %(page_size)d, includeClosedPrs: true) {
          pageInfo { hasNextPage }
          nodes {
            number
            merged
            title
            mergeCommit { parents(first: 1) { nodes { oid } } }
            files(first: %(file_page_size)d) {
              pageInfo { hasNextPage }
              nodes { path }
            }
          }
        }
      }
    }
  }
}
""" % {
    "page_size": LINKED_PR_PAGE_SIZE,
    "file_page_size": CHANGED_FILE_PAGE_SIZE,
}


def _base_commit(pr_node: dict[str, Any]) -> str:
    """The base branch immediately before the fix landed.

    That is the first parent of the merge commit. baseRefOid would give the
    base branch as it stands now, long after the fix, which is exactly the
    state the leakage control exists to avoid indexing.
    """
    merge_commit = pr_node.get("mergeCommit") or {}
    parents = (merge_commit.get("parents") or {}).get("nodes") or []
    if not parents:
        return ""
    return parents[0].get("oid") or ""


def parse_issues_page(
    payload: dict[str, Any],
) -> tuple[list[CandidateIssue], dict[int, CandidatePR], str | None]:
    container = payload["data"]["repository"]["issues"]
    issues: list[CandidateIssue] = []
    prs: dict[int, CandidatePR] = {}

    for node in container["nodes"]:
        linkage = node.get("closedByPullRequestsReferences") or {}
        pr_numbers: list[int] = []
        for pr_node in linkage.get("nodes") or []:
            number = pr_node["number"]
            pr_numbers.append(number)
            files = pr_node.get("files") or {}
            prs[number] = CandidatePR(
                number=number,
                merged=bool(pr_node.get("merged")),
                title=pr_node.get("title") or "",
                changed_files=tuple(f["path"] for f in files.get("nodes") or []),
                base_commit=_base_commit(pr_node),
                files_truncated=bool(
                    (files.get("pageInfo") or {}).get("hasNextPage")
                ),
            )
        issues.append(
            CandidateIssue(
                number=node["number"],
                title=node["title"],
                body=node.get("body") or "",
                labels=tuple(l["name"] for l in node["labels"]["nodes"]),
                linked_pr_numbers=tuple(pr_numbers),
                linkage_truncated=bool(
                    (linkage.get("pageInfo") or {}).get("hasNextPage")
                ),
            )
        )

    page_info = container["pageInfo"]
    cursor = page_info["endCursor"] if page_info["hasNextPage"] else None
    return issues, prs, cursor


class GitHubClient:
    def __init__(
        self,
        token: str,
        repo: str,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ) -> None:
        owner, name = repo.split("/", 1)
        self._owner = owner
        self._name = name
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"bearer {token}"},
            transport=transport,
            timeout=30.0,
        )

    async def _post_page(self, cursor: str | None) -> dict[str, Any]:
        """One GraphQL request, with bounded retry on transient failures."""
        last_errors: Sequence[Any] = [{"message": "no attempt was made"}]

        for attempt in range(self._max_retries + 1):
            if attempt:
                await asyncio.sleep(self._backoff_base * 2 ** (attempt - 1))

            response = await self._client.post(
                GITHUB_GRAPHQL_URL,
                json={
                    "query": ISSUES_QUERY,
                    "variables": {
                        "owner": self._owner,
                        "name": self._name,
                        "after": cursor,
                    },
                },
            )

            if response.status_code in RETRYABLE_STATUSES:
                last_errors = [
                    {
                        "type": "HTTP",
                        "message": f"HTTP {response.status_code} from GitHub",
                    }
                ]
                continue

            response.raise_for_status()
            payload = response.json()
            errors = payload.get("errors")
            if not errors:
                return payload

            last_errors = errors
            if not _is_retryable(errors):
                raise GitHubError(errors)

        raise GitHubError(last_errors)

    async def fetch_issues(
        self, limit: int
    ) -> tuple[list[CandidateIssue], dict[int, CandidatePR]]:
        issues: list[CandidateIssue] = []
        prs: dict[int, CandidatePR] = {}
        cursor: str | None = None

        while len(issues) < limit:
            payload = await self._post_page(cursor)
            page_issues, page_prs, cursor = parse_issues_page(payload)
            if not page_issues:
                # A page with no issues means no progress is being made even
                # if the server claims there is a next page; stop instead of
                # looping forever.
                break
            issues.extend(page_issues)
            prs.update(page_prs)
            if cursor is None:
                break

        return issues[:limit], prs

    async def aclose(self) -> None:
        await self._client.aclose()

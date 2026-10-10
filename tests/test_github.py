import asyncio
import json
from pathlib import Path

import httpx
import pytest

from miresolver.github import (
    ISSUES_QUERY,
    GitHubClient,
    GitHubError,
    parse_issues_page,
)

FIXTURE = Path(__file__).parent / "fixtures" / "issues_page.json"


def test_parse_issues_page_extracts_issue_and_linked_pr():
    payload = json.loads(FIXTURE.read_text())
    issues, prs, cursor = parse_issues_page(payload)

    assert len(issues) == 2
    issue = issues[0]
    assert issue.number == 4821
    assert issue.title == "crash when config file is empty"
    assert issue.labels == ("bug",)
    assert issue.linked_pr_numbers == (4830,)
    assert issue.linkage_truncated is False

    assert prs[4830].merged is True
    assert prs[4830].changed_files == ("src/config/parser.ts",)
    assert cursor == "Y3Vyc29yOnYyOpHOAA"


def test_query_uses_the_closing_relation_not_cross_references():
    """A cross-reference fires when an issue number is merely mentioned, so
    it cannot stand in for "this PR fixed this issue"."""
    assert "CROSS_REFERENCED_EVENT" not in ISSUES_QUERY
    assert "closedByPullRequestsReferences" in ISSUES_QUERY
    assert "includeClosedPrs: true" in ISSUES_QUERY


def test_parse_ignores_cross_reference_timeline_items():
    payload = json.loads(FIXTURE.read_text())
    node = payload["data"]["repository"]["issues"]["nodes"][0]
    node["timelineItems"] = {
        "nodes": [
            {
                "subject": {
                    "number": 9999,
                    "merged": True,
                    "title": "changelog for 1.90",
                    "files": {"nodes": [{"path": "CHANGELOG.md"}]},
                }
            }
        ]
    }
    issues, prs, _ = parse_issues_page(payload)
    assert issues[0].linked_pr_numbers == (4830,)
    assert 9999 not in prs


def test_parse_flags_truncated_linkage():
    payload = json.loads(FIXTURE.read_text())
    issues, _, _ = parse_issues_page(payload)
    assert issues[1].number == 4900
    assert issues[1].linkage_truncated is True


def test_parse_issues_page_returns_none_cursor_on_last_page():
    payload = json.loads(FIXTURE.read_text())
    payload["data"]["repository"]["issues"]["pageInfo"]["hasNextPage"] = False
    _, _, cursor = parse_issues_page(payload)
    assert cursor is None


@pytest.mark.timeout(5)
def test_fetch_issues_stops_on_empty_page_with_next_page_true():
    """Regression test: a page with hasNextPage=true but zero nodes must not
    loop forever. pytest-timeout enforces a hard, signal-based wall-clock
    bound at the process level, so a reintroduced infinite loop fails the
    test after 5s instead of hanging the suite (an asyncio-level timeout
    such as asyncio.wait_for is not sufficient here: a tight loop with no
    real I/O delay can starve the event loop and never yield to the
    cooperative cancellation)."""
    empty_page_with_next = {
        "data": {
            "repository": {
                "issues": {
                    "pageInfo": {"hasNextPage": True, "endCursor": "cursor"},
                    "nodes": [],
                }
            }
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=empty_page_with_next)

    async def run() -> tuple[list, dict]:
        transport = httpx.MockTransport(handler)
        client = GitHubClient(
            token="fake-token", repo="octocat/hello-world", transport=transport
        )
        try:
            return await client.fetch_issues(limit=10)
        finally:
            await client.aclose()

    issues, prs = asyncio.run(run())

    assert issues == []
    assert prs == {}


def test_parse_issues_page_extracts_the_pre_fix_base_commit():
    payload = json.loads(FIXTURE.read_text())
    _, prs, _ = parse_issues_page(payload)
    assert prs[4830].base_commit == "9f1c2ab7d3e4f50617283940abcdef1234567890"


def test_query_requests_the_merge_commit_first_parent():
    """The first parent of the merge commit is the base branch immediately
    before the fix landed. baseRefOid would be the base branch as it is
    today, which is not what gets indexed."""
    assert "mergeCommit" in ISSUES_QUERY
    assert "parents(first: 1)" in ISSUES_QUERY


def test_parse_tolerates_a_pull_request_with_no_merge_commit():
    payload = json.loads(FIXTURE.read_text())
    node = payload["data"]["repository"]["issues"]["nodes"][0]
    node["closedByPullRequestsReferences"]["nodes"][0]["mergeCommit"] = None
    _, prs, _ = parse_issues_page(payload)
    assert prs[4830].base_commit == ""


# --- transport and GraphQL error handling (IMPORTANT 4) ------------------


def _ok_page() -> dict:
    return {
        "data": {
            "repository": {
                "issues": {
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                    "nodes": [],
                }
            }
        }
    }


def _one_issue_page() -> dict:
    payload = json.loads(FIXTURE.read_text())
    payload["data"]["repository"]["issues"]["pageInfo"] = {
        "hasNextPage": False,
        "endCursor": None,
    }
    return payload


def _run(handler, **kw):
    async def go():
        client = GitHubClient(
            token="fake-token",
            repo="octocat/hello-world",
            transport=httpx.MockTransport(handler),
            backoff_base=0.0,
            **kw,
        )
        try:
            return await client.fetch_issues(limit=10)
        finally:
            await client.aclose()

    return asyncio.run(go())


def test_graphql_errors_are_raised_not_silently_parsed():
    """GitHub answers 200 with an errors block, so raise_for_status passes
    and the old code blew up on NoneType deep inside the parser."""
    payload = {
        "data": {"repository": None},
        "errors": [{"type": "NOT_FOUND", "message": "Could not resolve to a Repository"}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with pytest.raises(GitHubError) as exc:
        _run(handler)
    assert "Could not resolve to a Repository" in str(exc.value)


def test_partial_graphql_errors_do_not_yield_a_quietly_biased_corpus():
    """A timeout on one field returns data plus an errors block. Accepting
    that data means issues lose their linked PRs, get rejected as having no
    merged PR, and the run reports success on a skewed corpus."""
    payload = _one_issue_page()
    payload["errors"] = [
        {"type": "NOT_FOUND", "message": "timedout while resolving field"}
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with pytest.raises(GitHubError):
        _run(handler)


def test_rate_limited_response_is_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "data": None,
                    "errors": [{"type": "RATE_LIMITED", "message": "slow down"}],
                },
            )
        return httpx.Response(200, json=_ok_page())

    issues, _ = _run(handler)
    assert calls["n"] == 2
    assert issues == []


def test_server_error_status_is_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, json=_ok_page())

    issues, _ = _run(handler)
    assert calls["n"] == 3
    assert issues == []


def test_retries_are_bounded():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    with pytest.raises(GitHubError):
        _run(handler, max_retries=2)
    assert calls["n"] == 3


def test_client_error_is_not_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="bad credentials")

    with pytest.raises(httpx.HTTPStatusError):
        _run(handler)
    assert calls["n"] == 1

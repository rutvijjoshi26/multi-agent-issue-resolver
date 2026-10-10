# Issue Corpus and Answer Key Implementation Plan


**Goal:** Build a reproducible corpus of VS Code issues linked to the pull requests that fixed them, and from it produce a versioned evaluation manifest that later plans use as their answer key.

**Architecture:** A Python CLI fetches closed issues and their linked merged PRs from the GitHub GraphQL API, stores them in Postgres, applies pure-function inclusion rules to decide which issues are usable as evaluation cases, and writes a deterministic manifest file pinning the eval set. Database I/O stays thin so the rules that decide correctness are pure and unit-testable without a database.

**Tech Stack:** Python 3.12, httpx, psycopg 3, pytest, Docker Compose (Postgres 16), GitHub GraphQL API v4.

**Spec:** `docs/specs/2026-09-21-multi-agent-issue-resolver-design.md`

## Global Constraints

- Target repository for Phase 1: `microsoft/vscode`.
- Inclusion rules, applied exactly and documented in the README: issue has a bug label; issue links to exactly one merged PR; that PR touches between 1 and 5 source files; the PR is not a revert.
- Evaluation set is tiered: a 100-issue development subset that is a strict subset of the 500-issue published set.
- Every number the project reports must be reproducible from a committed manifest. The manifest pins issue numbers and the base commit for each, never "the latest N issues."
- No network calls in unit tests. GitHub responses are recorded as fixtures.
- Python 3.12 or newer. Type hints on all public functions.

---

### Task 1: Project scaffold and configuration

**Files:**
- Create: `pyproject.toml`
- Create: `docker-compose.yml`
- Create: `src/miresolver/__init__.py`
- Create: `src/miresolver/config.py`
- Create: `.env.example`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `load_config(env: Mapping[str, str]) -> Config` where `Config` is a frozen dataclass with fields `github_token: str`, `database_url: str`, `target_repo: str`. Raises `ConfigError` on a missing required key.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
from miresolver.config import load_config, Config, ConfigError


def test_load_config_reads_required_values():
    cfg = load_config({
        "GITHUB_TOKEN": "ghp_example",
        "DATABASE_URL": "postgresql://localhost/test",
        "TARGET_REPO": "microsoft/vscode",
    })
    assert cfg == Config(
        github_token="ghp_example",
        database_url="postgresql://localhost/test",
        target_repo="microsoft/vscode",
    )


def test_load_config_defaults_target_repo():
    cfg = load_config({
        "GITHUB_TOKEN": "ghp_example",
        "DATABASE_URL": "postgresql://localhost/test",
    })
    assert cfg.target_repo == "microsoft/vscode"


def test_load_config_raises_on_missing_token():
    with pytest.raises(ConfigError, match="GITHUB_TOKEN"):
        load_config({"DATABASE_URL": "postgresql://localhost/test"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'miresolver.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/miresolver/config.py
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_TARGET_REPO = "microsoft/vscode"


class ConfigError(Exception):
    """Raised when required configuration is absent."""


@dataclass(frozen=True)
class Config:
    github_token: str
    database_url: str
    target_repo: str


def _required(env: Mapping[str, str], key: str) -> str:
    try:
        return env[key]
    except KeyError as exc:
        raise ConfigError(f"{key} is required") from exc


def load_config(env: Mapping[str, str]) -> Config:
    return Config(
        github_token=_required(env, "GITHUB_TOKEN"),
        database_url=_required(env, "DATABASE_URL"),
        target_repo=env.get("TARGET_REPO", DEFAULT_TARGET_REPO),
    )
```

```toml
# pyproject.toml
[project]
name = "miresolver"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["httpx>=0.27", "psycopg[binary]>=3.2"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/miresolver"]

[tool.pytest.ini_options]
pythonpath = ["src"]
```

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: miresolver
      POSTGRES_PASSWORD: miresolver
      POSTGRES_DB: miresolver
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]
volumes:
  pgdata:
```

```bash
# .env.example
GITHUB_TOKEN=ghp_replace_me
DATABASE_URL=postgresql://miresolver:miresolver@localhost:5432/miresolver
TARGET_REPO=microsoft/vscode
```

```python
# src/miresolver/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml docker-compose.yml .env.example src/miresolver tests/test_config.py
git commit -m "feat: project scaffold and configuration loading"
```

---

### Task 2: Inclusion rules as pure functions

Written before the GitHub client on purpose. These rules decide what counts as a valid evaluation case, so they are the part most worth getting right and the easiest to test in isolation.

**Files:**
- Create: `src/miresolver/rules.py`
- Test: `tests/test_rules.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `@dataclass(frozen=True) class CandidateIssue` with fields `number: int`, `title: str`, `body: str`, `labels: tuple[str, ...]`, `linked_pr_numbers: tuple[int, ...]`
  - `@dataclass(frozen=True) class CandidatePR` with fields `number: int`, `merged: bool`, `title: str`, `changed_files: tuple[str, ...]`
  - `is_eligible(issue: CandidateIssue, prs: Mapping[int, CandidatePR]) -> Eligibility`
  - `@dataclass(frozen=True) class Eligibility` with fields `eligible: bool`, `reason: str`. `reason` is `"ok"` when eligible, otherwise one of `"no_bug_label"`, `"not_exactly_one_merged_pr"`, `"file_count_out_of_range"`, `"revert"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rules.py
from miresolver.rules import CandidateIssue, CandidatePR, is_eligible

BUG = ("bug",)


def issue(**kw) -> CandidateIssue:
    base = dict(number=1, title="crash on empty config",
                body="it crashes", labels=BUG, linked_pr_numbers=(10,))
    base.update(kw)
    return CandidateIssue(**base)


def pr(**kw) -> CandidatePR:
    base = dict(number=10, merged=True, title="fix parser crash",
                changed_files=("src/config/parser.ts",))
    base.update(kw)
    return CandidatePR(**base)


def test_eligible_when_all_rules_pass():
    result = is_eligible(issue(), {10: pr()})
    assert result.eligible is True
    assert result.reason == "ok"


def test_rejects_issue_without_bug_label():
    result = is_eligible(issue(labels=("feature-request",)), {10: pr()})
    assert result.eligible is False
    assert result.reason == "no_bug_label"


def test_rejects_issue_with_two_linked_prs():
    result = is_eligible(
        issue(linked_pr_numbers=(10, 11)),
        {10: pr(), 11: pr(number=11)},
    )
    assert result.reason == "not_exactly_one_merged_pr"


def test_rejects_issue_whose_only_pr_is_unmerged():
    result = is_eligible(issue(), {10: pr(merged=False)})
    assert result.reason == "not_exactly_one_merged_pr"


def test_counts_only_merged_prs_toward_the_one_pr_rule():
    result = is_eligible(
        issue(linked_pr_numbers=(10, 11)),
        {10: pr(), 11: pr(number=11, merged=False)},
    )
    assert result.eligible is True


def test_rejects_pr_touching_zero_files():
    result = is_eligible(issue(), {10: pr(changed_files=())})
    assert result.reason == "file_count_out_of_range"


def test_rejects_pr_touching_six_files():
    files = tuple(f"src/f{i}.ts" for i in range(6))
    result = is_eligible(issue(), {10: pr(changed_files=files)})
    assert result.reason == "file_count_out_of_range"


def test_accepts_pr_touching_five_files():
    files = tuple(f"src/f{i}.ts" for i in range(5))
    result = is_eligible(issue(), {10: pr(changed_files=files)})
    assert result.eligible is True


def test_rejects_revert_pr():
    result = is_eligible(issue(), {10: pr(title='Revert "fix parser crash"')})
    assert result.reason == "revert"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rules.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'miresolver.rules'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/miresolver/rules.py
from collections.abc import Mapping
from dataclasses import dataclass

MIN_CHANGED_FILES = 1
MAX_CHANGED_FILES = 5
BUG_LABELS = frozenset({"bug"})


@dataclass(frozen=True)
class CandidateIssue:
    number: int
    title: str
    body: str
    labels: tuple[str, ...]
    linked_pr_numbers: tuple[int, ...]


@dataclass(frozen=True)
class CandidatePR:
    number: int
    merged: bool
    title: str
    changed_files: tuple[str, ...]


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reason: str


def _is_revert(pr: CandidatePR) -> bool:
    return pr.title.strip().lower().startswith("revert")


def is_eligible(
    issue: CandidateIssue, prs: Mapping[int, CandidatePR]
) -> Eligibility:
    if not BUG_LABELS.intersection(label.lower() for label in issue.labels):
        return Eligibility(False, "no_bug_label")

    merged = [
        prs[n] for n in issue.linked_pr_numbers if n in prs and prs[n].merged
    ]
    if len(merged) != 1:
        return Eligibility(False, "not_exactly_one_merged_pr")

    pr = merged[0]
    if _is_revert(pr):
        return Eligibility(False, "revert")

    if not MIN_CHANGED_FILES <= len(pr.changed_files) <= MAX_CHANGED_FILES:
        return Eligibility(False, "file_count_out_of_range")

    return Eligibility(True, "ok")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rules.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/miresolver/rules.py tests/test_rules.py
git commit -m "feat: inclusion rules for evaluation candidates"
```

---

### Task 3: GitHub GraphQL client

**Files:**
- Create: `src/miresolver/github.py`
- Create: `tests/fixtures/issues_page.json`
- Test: `tests/test_github.py`

**Interfaces:**
- Consumes: `Config` from Task 1; `CandidateIssue`, `CandidatePR` from Task 2
- Produces:
  - `ISSUES_QUERY: str` (GraphQL document)
  - `parse_issues_page(payload: dict) -> tuple[list[CandidateIssue], dict[int, CandidatePR], str | None]` returning issues, PRs keyed by number, and the next cursor or `None`
  - `class GitHubClient` with `__init__(self, token: str, repo: str, transport: httpx.AsyncBaseTransport | None = None)` and `async def fetch_issues(self, limit: int) -> tuple[list[CandidateIssue], dict[int, CandidatePR]]`

Parsing is a pure function so it is testable without a network. The client only handles paging and transport.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_github.py
import json
from pathlib import Path

from miresolver.github import parse_issues_page

FIXTURE = Path(__file__).parent / "fixtures" / "issues_page.json"


def test_parse_issues_page_extracts_issue_and_linked_pr():
    payload = json.loads(FIXTURE.read_text())
    issues, prs, cursor = parse_issues_page(payload)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.number == 4821
    assert issue.title == "crash when config file is empty"
    assert issue.labels == ("bug",)
    assert issue.linked_pr_numbers == (4830,)

    assert prs[4830].merged is True
    assert prs[4830].changed_files == ("src/config/parser.ts",)
    assert cursor == "Y3Vyc29yOnYyOpHOAA"


def test_parse_issues_page_returns_none_cursor_on_last_page():
    payload = json.loads(FIXTURE.read_text())
    payload["data"]["repository"]["issues"]["pageInfo"]["hasNextPage"] = False
    _, _, cursor = parse_issues_page(payload)
    assert cursor is None
```

```json
// tests/fixtures/issues_page.json
{
  "data": {
    "repository": {
      "issues": {
        "pageInfo": { "hasNextPage": true, "endCursor": "Y3Vyc29yOnYyOpHOAA" },
        "nodes": [
          {
            "number": 4821,
            "title": "crash when config file is empty",
            "body": "Opening an empty settings file crashes the window.",
            "labels": { "nodes": [{ "name": "bug" }] },
            "timelineItems": {
              "nodes": [
                {
                  "subject": {
                    "number": 4830,
                    "merged": true,
                    "title": "fix parser crash on empty input",
                    "files": { "nodes": [{ "path": "src/config/parser.ts" }] }
                  }
                }
              ]
            }
          }
        ]
      }
    }
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_github.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'miresolver.github'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/miresolver/github.py
from typing import Any

import httpx

from miresolver.rules import CandidateIssue, CandidatePR

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

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
        timelineItems(first: 20, itemTypes: [CLOSED_EVENT, CROSS_REFERENCED_EVENT]) {
          nodes {
            ... on CrossReferencedEvent {
              subject {
                ... on PullRequest {
                  number
                  merged
                  title
                  files(first: 20) { nodes { path } }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""


def parse_issues_page(
    payload: dict[str, Any],
) -> tuple[list[CandidateIssue], dict[int, CandidatePR], str | None]:
    container = payload["data"]["repository"]["issues"]
    issues: list[CandidateIssue] = []
    prs: dict[int, CandidatePR] = {}

    for node in container["nodes"]:
        pr_numbers: list[int] = []
        for item in node["timelineItems"]["nodes"]:
            subject = item.get("subject") or {}
            if "number" not in subject:
                continue
            number = subject["number"]
            pr_numbers.append(number)
            prs[number] = CandidatePR(
                number=number,
                merged=bool(subject.get("merged")),
                title=subject.get("title", ""),
                changed_files=tuple(
                    f["path"] for f in subject.get("files", {}).get("nodes", [])
                ),
            )
        issues.append(
            CandidateIssue(
                number=node["number"],
                title=node["title"],
                body=node.get("body") or "",
                labels=tuple(l["name"] for l in node["labels"]["nodes"]),
                linked_pr_numbers=tuple(pr_numbers),
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
    ) -> None:
        owner, name = repo.split("/", 1)
        self._owner = owner
        self._name = name
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"bearer {token}"},
            transport=transport,
            timeout=30.0,
        )

    async def fetch_issues(
        self, limit: int
    ) -> tuple[list[CandidateIssue], dict[int, CandidatePR]]:
        issues: list[CandidateIssue] = []
        prs: dict[int, CandidatePR] = {}
        cursor: str | None = None

        while len(issues) < limit:
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
            response.raise_for_status()
            page_issues, page_prs, cursor = parse_issues_page(response.json())
            issues.extend(page_issues)
            prs.update(page_prs)
            if cursor is None:
                break

        return issues[:limit], prs

    async def aclose(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_github.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/miresolver/github.py tests/test_github.py tests/fixtures/issues_page.json
git commit -m "feat: GitHub GraphQL client and page parsing"
```

---

### Task 4: Database schema and storage

**Files:**
- Create: `src/miresolver/schema.sql`
- Create: `src/miresolver/store.py`
- Test: `tests/test_store.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: `CandidateIssue`, `CandidatePR` from Task 2
- Produces:
  - `create_schema(conn: psycopg.Connection) -> None`
  - `save_issues(conn, issues: Sequence[CandidateIssue], prs: Mapping[int, CandidatePR]) -> None` (idempotent, upserts on primary key)
  - `load_issue_numbers(conn) -> list[int]` returning ascending issue numbers

Requires Postgres running via `docker compose up -d postgres`. Tests are skipped when `TEST_DATABASE_URL` is unset so the suite stays green without Docker.

- [ ] **Step 1: Write the failing test**

```python
# tests/conftest.py
import os

import psycopg
import pytest


@pytest.fixture
def conn():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    with psycopg.connect(url, autocommit=True) as connection:
        with connection.cursor() as cur:
            cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        yield connection
```

```python
# tests/test_store.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose up -d postgres && TEST_DATABASE_URL=postgresql://miresolver:miresolver@localhost:5432/miresolver pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'miresolver.store'`

- [ ] **Step 3: Write minimal implementation**

```sql
-- src/miresolver/schema.sql
CREATE TABLE IF NOT EXISTS issues (
    number       INTEGER PRIMARY KEY,
    title        TEXT NOT NULL,
    body         TEXT NOT NULL,
    labels       TEXT[] NOT NULL
);

CREATE TABLE IF NOT EXISTS pull_requests (
    number        INTEGER PRIMARY KEY,
    merged        BOOLEAN NOT NULL,
    title         TEXT NOT NULL,
    changed_files TEXT[] NOT NULL
);

CREATE TABLE IF NOT EXISTS issue_pr_links (
    issue_number INTEGER NOT NULL REFERENCES issues(number) ON DELETE CASCADE,
    pr_number    INTEGER NOT NULL REFERENCES pull_requests(number) ON DELETE CASCADE,
    PRIMARY KEY (issue_number, pr_number)
);
```

```python
# src/miresolver/store.py
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
                INSERT INTO pull_requests (number, merged, title, changed_files)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (number) DO UPDATE SET
                    merged = EXCLUDED.merged,
                    title = EXCLUDED.title,
                    changed_files = EXCLUDED.changed_files
                """,
                (pr.number, pr.merged, pr.title, list(pr.changed_files)),
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
```

Add to `pyproject.toml` under `[tool.hatch.build.targets.wheel]`:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/miresolver/schema.sql" = "miresolver/schema.sql"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TEST_DATABASE_URL=postgresql://miresolver:miresolver@localhost:5432/miresolver pytest tests/test_store.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/miresolver/schema.sql src/miresolver/store.py tests/test_store.py tests/conftest.py pyproject.toml
git commit -m "feat: Postgres schema and issue storage"
```

---

### Task 5: Evaluation manifest builder

The manifest is what makes every published number reproducible. It pins exact issue numbers and the base commit each one must be evaluated against.

**Files:**
- Create: `src/miresolver/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `CandidateIssue`, `CandidatePR`, `is_eligible` from Task 2
- Produces:
  - `@dataclass(frozen=True) class EvalCase` with fields `issue_number: int`, `pr_number: int`, `changed_files: tuple[str, ...]`
  - `build_manifest(issues, prs, *, dev_size: int, full_size: int, seed: int) -> Manifest`
  - `@dataclass(frozen=True) class Manifest` with fields `dev: tuple[EvalCase, ...]`, `full: tuple[EvalCase, ...]`, `rejected: Mapping[str, int]`
  - `manifest_to_json(manifest: Manifest) -> str` producing key-sorted, deterministic JSON

`dev` must be a strict subset of `full`. Selection is seeded so two runs over the same input produce identical manifests.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
import json

from miresolver.manifest import build_manifest, manifest_to_json
from miresolver.rules import CandidateIssue, CandidatePR


def make_pair(n: int, *, labels=("bug",), merged=True, files=None):
    files = files if files is not None else (f"src/f{n}.ts",)
    issue = CandidateIssue(
        number=n, title=f"issue {n}", body="body",
        labels=labels, linked_pr_numbers=(1000 + n,),
    )
    pr = CandidatePR(
        number=1000 + n, merged=merged, title=f"fix {n}", changed_files=files,
    )
    return issue, pr


def build(count: int, **kw):
    issues, prs = [], {}
    for n in range(1, count + 1):
        issue, pr = make_pair(n, **kw)
        issues.append(issue)
        prs[pr.number] = pr
    return issues, prs


def test_dev_set_is_strict_subset_of_full_set():
    issues, prs = build(20)
    m = build_manifest(issues, prs, dev_size=5, full_size=10, seed=42)
    assert len(m.dev) == 5
    assert len(m.full) == 10
    dev_numbers = {c.issue_number for c in m.dev}
    full_numbers = {c.issue_number for c in m.full}
    assert dev_numbers < full_numbers


def test_selection_is_deterministic_for_a_seed():
    issues, prs = build(20)
    a = build_manifest(issues, prs, dev_size=5, full_size=10, seed=42)
    b = build_manifest(issues, prs, dev_size=5, full_size=10, seed=42)
    assert manifest_to_json(a) == manifest_to_json(b)


def test_ineligible_issues_are_excluded_and_counted():
    issues, prs = build(10)
    bad_issue, bad_pr = make_pair(99, labels=("feature-request",))
    issues.append(bad_issue)
    prs[bad_pr.number] = bad_pr

    m = build_manifest(issues, prs, dev_size=2, full_size=5, seed=1)
    assert 99 not in {c.issue_number for c in m.full}
    assert m.rejected["no_bug_label"] == 1


def test_manifest_records_the_changed_files_as_the_answer_key():
    issues, prs = build(5)
    m = build_manifest(issues, prs, dev_size=1, full_size=2, seed=7)
    case = m.full[0]
    assert case.changed_files == prs[case.pr_number].changed_files


def test_manifest_json_is_sorted_and_stable():
    issues, prs = build(10)
    m = build_manifest(issues, prs, dev_size=2, full_size=4, seed=3)
    parsed = json.loads(manifest_to_json(m))
    numbers = [c["issue_number"] for c in parsed["full"]]
    assert numbers == sorted(numbers)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'miresolver.manifest'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/miresolver/manifest.py
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from miresolver.rules import CandidateIssue, CandidatePR, is_eligible


@dataclass(frozen=True)
class EvalCase:
    issue_number: int
    pr_number: int
    changed_files: tuple[str, ...]


@dataclass(frozen=True)
class Manifest:
    dev: tuple[EvalCase, ...]
    full: tuple[EvalCase, ...]
    rejected: Mapping[str, int]


def build_manifest(
    issues: Sequence[CandidateIssue],
    prs: Mapping[int, CandidatePR],
    *,
    dev_size: int,
    full_size: int,
    seed: int,
) -> Manifest:
    if dev_size > full_size:
        raise ValueError("dev_size must not exceed full_size")

    eligible: list[EvalCase] = []
    rejected: Counter[str] = Counter()

    for issue in sorted(issues, key=lambda i: i.number):
        verdict = is_eligible(issue, prs)
        if not verdict.eligible:
            rejected[verdict.reason] += 1
            continue
        pr = next(
            prs[n] for n in issue.linked_pr_numbers if n in prs and prs[n].merged
        )
        eligible.append(
            EvalCase(
                issue_number=issue.number,
                pr_number=pr.number,
                changed_files=pr.changed_files,
            )
        )

    rng = random.Random(seed)
    full = sorted(
        rng.sample(eligible, min(full_size, len(eligible))),
        key=lambda c: c.issue_number,
    )
    dev = sorted(
        rng.sample(full, min(dev_size, len(full))),
        key=lambda c: c.issue_number,
    )
    return Manifest(dev=tuple(dev), full=tuple(full), rejected=dict(rejected))


def _case_to_dict(case: EvalCase) -> dict:
    return {
        "issue_number": case.issue_number,
        "pr_number": case.pr_number,
        "changed_files": list(case.changed_files),
    }


def manifest_to_json(manifest: Manifest) -> str:
    return json.dumps(
        {
            "dev": [_case_to_dict(c) for c in manifest.dev],
            "full": [_case_to_dict(c) for c in manifest.full],
            "rejected": dict(sorted(manifest.rejected.items())),
        },
        indent=2,
        sort_keys=True,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_manifest.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/miresolver/manifest.py tests/test_manifest.py
git commit -m "feat: deterministic evaluation manifest builder"
```

---

### Task 6: Ingestion CLI

**Files:**
- Create: `src/miresolver/cli.py`
- Modify: `pyproject.toml` (add console script entry point)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1 through 5
- Produces: console command `miresolver ingest --limit N --out PATH`, and `async def run_ingest(config: Config, *, limit: int, out_path: Path, dev_size: int, full_size: int, seed: int, client=None, conn=None) -> Manifest` for testing with injected dependencies.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import asyncio
import json
from pathlib import Path

from miresolver.cli import run_ingest
from miresolver.config import Config
from miresolver.rules import CandidateIssue, CandidatePR


class FakeClient:
    def __init__(self, issues, prs):
        self._issues = issues
        self._prs = prs

    async def fetch_issues(self, limit):
        return self._issues[:limit], self._prs

    async def aclose(self):
        return None


def test_run_ingest_writes_manifest_file(tmp_path: Path):
    issues, prs = [], {}
    for n in range(1, 11):
        issues.append(
            CandidateIssue(
                number=n, title=f"t{n}", body="b",
                labels=("bug",), linked_pr_numbers=(1000 + n,),
            )
        )
        prs[1000 + n] = CandidatePR(
            number=1000 + n, merged=True, title=f"fix {n}",
            changed_files=(f"src/f{n}.ts",),
        )

    out = tmp_path / "manifest.json"
    cfg = Config(github_token="x", database_url="", target_repo="microsoft/vscode")

    manifest = asyncio.run(
        run_ingest(
            cfg, limit=10, out_path=out, dev_size=2, full_size=4, seed=5,
            client=FakeClient(issues, prs), conn=None,
        )
    )

    assert out.exists()
    written = json.loads(out.read_text())
    assert len(written["full"]) == 4
    assert len(written["dev"]) == 2
    assert len(manifest.full) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'miresolver.cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/miresolver/cli.py
import argparse
import asyncio
import os
from pathlib import Path

import psycopg

from miresolver.config import Config, load_config
from miresolver.github import GitHubClient
from miresolver.manifest import Manifest, build_manifest, manifest_to_json
from miresolver.store import create_schema, save_issues


async def run_ingest(
    config: Config,
    *,
    limit: int,
    out_path: Path,
    dev_size: int,
    full_size: int,
    seed: int,
    client=None,
    conn=None,
) -> Manifest:
    owns_client = client is None
    if client is None:
        client = GitHubClient(config.github_token, config.target_repo)

    try:
        issues, prs = await client.fetch_issues(limit)
    finally:
        if owns_client:
            await client.aclose()

    if conn is not None:
        create_schema(conn)
        save_issues(conn, issues, prs)

    manifest = build_manifest(
        issues, prs, dev_size=dev_size, full_size=full_size, seed=seed
    )
    out_path.write_text(manifest_to_json(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(prog="miresolver")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="fetch issues and build the manifest")
    ingest.add_argument("--limit", type=int, default=2000)
    ingest.add_argument("--out", type=Path, default=Path("eval/manifest.json"))
    ingest.add_argument("--dev-size", type=int, default=100)
    ingest.add_argument("--full-size", type=int, default=500)
    ingest.add_argument("--seed", type=int, default=20260921)

    args = parser.parse_args()
    config = load_config(os.environ)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(config.database_url, autocommit=True) as conn:
        manifest = asyncio.run(
            run_ingest(
                config,
                limit=args.limit,
                out_path=args.out,
                dev_size=args.dev_size,
                full_size=args.full_size,
                seed=args.seed,
                conn=conn,
            )
        )

    print(f"eligible cases: {len(manifest.full)}")
    for reason, count in sorted(manifest.rejected.items()):
        print(f"  rejected {reason}: {count}")


if __name__ == "__main__":
    main()
```

Add to `pyproject.toml`:

```toml
[project.scripts]
miresolver = "miresolver.cli:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/miresolver/cli.py tests/test_cli.py pyproject.toml
git commit -m "feat: ingestion CLI producing the evaluation manifest"
```

---

### Task 7: README documenting the answer key rules

The spec requires the inclusion rules to be public rather than quietly applied. This task makes that true.

**Files:**
- Create: `README.md`
- Test: `tests/test_readme.py`

**Interfaces:**
- Consumes: rule constants from Task 2
- Produces: nothing consumed by later tasks

- [ ] **Step 1: Write the failing test**

```python
# tests/test_readme.py
from pathlib import Path

from miresolver.rules import MAX_CHANGED_FILES, MIN_CHANGED_FILES

README = Path(__file__).parent.parent / "README.md"


def test_readme_documents_every_inclusion_rule():
    text = README.read_text().lower()
    assert "bug label" in text
    assert "exactly one" in text and "merged" in text
    assert f"{MIN_CHANGED_FILES} and {MAX_CHANGED_FILES}" in text
    assert "revert" in text


def test_readme_states_the_leakage_control():
    text = README.read_text().lower()
    assert "before the fix" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_readme.py -v`
Expected: FAIL with `FileNotFoundError` for README.md

- [ ] **Step 3: Write minimal implementation**

````markdown
# Multi-Agent Issue Resolver

When a new issue arrives, this finds similar issues that were already solved,
investigates each candidate in parallel against the current code, and suggests
an approach with citations to the past issues and pull requests it relied on.
It abstains when nothing matches confidently.

## How results are measured

Accuracy is measured against real merged pull requests, so the answer key was
written by the maintainers who actually fixed each bug, not by us.

### Which issues count as evaluation cases

An issue is included only if all of the following hold:

- It carries a **bug label**
- It links to **exactly one merged** pull request
- That pull request touches between **1 and 5** source files, which excludes
  mass reformatting and sweeping refactors
- The pull request is not a **revert**

Rejected issues are counted by reason and reported alongside every run.

### Leakage control

For each evaluation case the repository is indexed at the commit immediately
**before the fix** landed. Indexing the current repository would mean the fix is
already present in the code, which would inflate every number and make the
result meaningless.

### Reproducibility

The evaluation set is pinned in a committed manifest listing exact issue
numbers and their answer keys. Selection is seeded, so rebuilding the manifest
from the same input produces an identical file.

## Setup

```bash
docker compose up -d postgres
pip install -e ".[dev]"
cp .env.example .env   # add your GitHub token
miresolver ingest --limit 2000 --out eval/manifest.json
```
````

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_readme.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_readme.py
git commit -m "docs: README documenting answer key rules and leakage control"
```

---

## Done when

- `pytest` passes with Postgres running, and passes with store tests skipped when it is not
- `miresolver ingest` produces `eval/manifest.json` with 500 cases in `full` and 100 in `dev`
- Rebuilding the manifest from the same input yields a byte-identical file
- README documents every inclusion rule and the leakage control

## What the next plan consumes from this one

- `eval/manifest.json`, the pinned evaluation set and answer key
- `issues`, `pull_requests`, and `issue_pr_links` tables in Postgres
- `CandidateIssue` and `CandidatePR` as the shared issue representation

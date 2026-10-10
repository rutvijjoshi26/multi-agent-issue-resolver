import asyncio
import json
from pathlib import Path

from miresolver import cli
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


def _make_issues_and_prs(count: int = 10):
    issues, prs = [], {}
    for n in range(1, count + 1):
        issues.append(
            CandidateIssue(
                number=n, title=f"t{n}", body="b",
                labels=("bug",), linked_pr_numbers=(1000 + n,),
            )
        )
        prs[1000 + n] = CandidatePR(
            number=1000 + n, merged=True, title=f"fix {n}",
            changed_files=(f"src/f{n}.ts",), base_commit=f"{n:040x}",
        )
    return issues, prs


def test_run_ingest_writes_manifest_file(tmp_path: Path):
    issues, prs = _make_issues_and_prs()

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


def test_main_reports_value_error_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    class EmptyClient:
        async def fetch_issues(self, limit):
            return [], {}

        async def aclose(self):
            return None

    monkeypatch.setattr(cli, "GitHubClient", lambda token, repo: EmptyClient())
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")

    out = tmp_path / "manifest.json"

    exit_code = cli.main(
        [
            "ingest",
            "--skip-db",
            "--out", str(out),
            "--dev-size", "5",
            "--full-size", "5",
        ]
    )

    assert exit_code == 1
    assert not out.exists()
    err = capsys.readouterr().err
    assert "ingest failed" in err


def test_skip_db_flag_avoids_database_connection(tmp_path, monkeypatch):
    issues, prs = _make_issues_and_prs()

    monkeypatch.setattr(cli, "GitHubClient", lambda token, repo: FakeClient(issues, prs))

    def fail_connect(*args, **kwargs):
        raise AssertionError("psycopg.connect should not be called when --skip-db is set")

    monkeypatch.setattr(cli.psycopg, "connect", fail_connect)
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")

    out = tmp_path / "manifest.json"

    exit_code = cli.main(
        [
            "ingest",
            "--skip-db",
            "--out", str(out),
            "--dev-size", "2",
            "--full-size", "4",
        ]
    )

    assert exit_code == 0
    assert out.exists()
    written = json.loads(out.read_text())
    assert len(written["full"]) == 4


def test_run_ingest_records_provenance_in_the_manifest(tmp_path: Path):
    issues, prs = _make_issues_and_prs()
    out = tmp_path / "manifest.json"
    cfg = Config(github_token="x", database_url="", target_repo="octo/repo")

    asyncio.run(
        run_ingest(
            cfg, limit=10, out_path=out, dev_size=2, full_size=4, seed=5,
            client=FakeClient(issues, prs), conn=None,
        )
    )

    meta = json.loads(out.read_text())["meta"]
    assert meta["repo"] == "octo/repo"
    assert meta["seed"] == 5
    assert meta["limit"] == 10
    assert meta["total_eligible"] == 10
    assert meta["full_size"] == 4


def test_summary_distinguishes_eligible_from_selected(tmp_path, monkeypatch, capsys):
    issues, prs = _make_issues_and_prs()
    monkeypatch.setattr(cli, "GitHubClient", lambda token, repo: FakeClient(issues, prs))
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")

    out = tmp_path / "manifest.json"
    assert cli.main(
        ["ingest", "--skip-db", "--out", str(out),
         "--dev-size", "2", "--full-size", "4"]
    ) == 0

    printed = capsys.readouterr().out
    assert "eligible cases: 10" in printed
    assert "selected: full=4, dev=2" in printed


def test_undersized_pool_fails_loudly_by_default(tmp_path, monkeypatch, capsys):
    issues, prs = _make_issues_and_prs(count=6)
    monkeypatch.setattr(cli, "GitHubClient", lambda token, repo: FakeClient(issues, prs))
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")

    out = tmp_path / "manifest.json"
    assert cli.main(
        ["ingest", "--skip-db", "--out", str(out),
         "--dev-size", "2", "--full-size", "20"]
    ) == 1
    assert not out.exists()
    assert "only 6 eligible" in capsys.readouterr().err


def test_allow_undersized_writes_a_labelled_manifest(tmp_path, monkeypatch, capsys):
    issues, prs = _make_issues_and_prs(count=6)
    monkeypatch.setattr(cli, "GitHubClient", lambda token, repo: FakeClient(issues, prs))
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")

    out = tmp_path / "manifest.json"
    assert cli.main(
        ["ingest", "--skip-db", "--out", str(out), "--allow-undersized",
         "--dev-size", "2", "--full-size", "20"]
    ) == 0
    written = json.loads(out.read_text())
    assert written["meta"]["undersized"] is True
    assert len(written["full"]) == 6
    assert "UNDERSIZED" in capsys.readouterr().err


def test_skip_db_does_not_require_a_database_url(tmp_path, monkeypatch):
    issues, prs = _make_issues_and_prs()
    monkeypatch.setattr(cli, "GitHubClient", lambda token, repo: FakeClient(issues, prs))
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(cli, "read_dotenv", lambda path=None: {})

    out = tmp_path / "manifest.json"
    assert cli.main(
        ["ingest", "--skip-db", "--out", str(out),
         "--dev-size", "2", "--full-size", "4"]
    ) == 0


def test_missing_config_exits_one_with_a_message(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(cli, "read_dotenv", lambda path=None: {})

    out = tmp_path / "manifest.json"
    assert cli.main(["ingest", "--skip-db", "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "GITHUB_TOKEN" in err
    assert "Traceback" not in err


def test_dotenv_file_supplies_configuration(tmp_path, monkeypatch):
    issues, prs = _make_issues_and_prs()
    monkeypatch.setattr(cli, "GitHubClient", lambda token, repo: FakeClient(issues, prs))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("GITHUB_TOKEN=from-dotenv\nTARGET_REPO=octo/repo\n")
    monkeypatch.chdir(tmp_path)

    out = tmp_path / "manifest.json"
    assert cli.main(
        ["ingest", "--skip-db", "--out", str(out),
         "--dev-size", "2", "--full-size", "4"]
    ) == 0
    assert json.loads(out.read_text())["meta"]["repo"] == "octo/repo"

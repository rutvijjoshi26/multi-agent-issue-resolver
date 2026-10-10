import asyncio
import subprocess

import httpx
import pytest

from miresolver.code.reader import (
    CodeReadError,
    GitHubRawCodeReader,
    LocalGitCodeReader,
)

SHA = "a" * 40


def _raw_reader(tmp_path, handler, calls):
    def counting(request):
        calls.append(str(request.url))
        return handler(request)

    return GitHubRawCodeReader(
        "owner/repo", tmp_path / "cache", transport=httpx.MockTransport(counting)
    )


def test_raw_reader_fetches_the_file_at_the_given_commit(tmp_path):
    calls: list[str] = []
    reader = _raw_reader(tmp_path, lambda r: httpx.Response(200, text="code"), calls)
    text = asyncio.run(reader.read(SHA, "src/a.ts"))
    asyncio.run(reader.aclose())
    assert text == "code"
    assert calls == [f"https://raw.githubusercontent.com/owner/repo/{SHA}/src/a.ts"]


def test_raw_reader_returns_none_for_a_file_missing_at_that_commit(tmp_path):
    calls: list[str] = []
    reader = _raw_reader(tmp_path, lambda r: httpx.Response(404), calls)
    assert asyncio.run(reader.read(SHA, "gone.ts")) is None
    asyncio.run(reader.aclose())


def test_raw_reader_serves_hits_and_misses_from_the_cache(tmp_path):
    calls: list[str] = []

    def handler(request):
        if request.url.path.endswith("/a.ts"):
            return httpx.Response(200, text="code")
        return httpx.Response(404)

    reader = _raw_reader(tmp_path, handler, calls)
    for _ in range(2):
        assert asyncio.run(reader.read(SHA, "a.ts")) == "code"
        assert asyncio.run(reader.read(SHA, "b.ts")) is None
    asyncio.run(reader.aclose())
    assert len(calls) == 2


def test_raw_reader_raises_on_server_errors(tmp_path):
    reader = _raw_reader(tmp_path, lambda r: httpx.Response(503), [])
    with pytest.raises(CodeReadError):
        asyncio.run(reader.read(SHA, "a.ts"))
    asyncio.run(reader.aclose())


@pytest.mark.parametrize(
    "commit,path",
    [(SHA, "../etc/passwd"), (SHA, "/abs.ts"), ("main; rm -rf", "a.ts"), ("", "a.ts")],
)
def test_unsafe_commits_and_paths_are_rejected(tmp_path, commit, path):
    reader = _raw_reader(tmp_path, lambda r: httpx.Response(200, text="x"), [])
    with pytest.raises(CodeReadError):
        asyncio.run(reader.read(commit, path))
    asyncio.run(reader.aclose())


def _git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_local_reader_returns_content_as_of_the_older_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.ts").write_text("before the fix")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "one")
    old = _git(repo, "rev-parse", "HEAD")
    (repo / "a.ts").write_text("after the fix")
    _git(repo, "commit", "-qam", "two")

    reader = LocalGitCodeReader(repo)
    assert asyncio.run(reader.read(old, "a.ts")) == "before the fix"
    assert asyncio.run(reader.read(old, "missing.ts")) is None


def test_raw_cache_is_keyed_by_repo_so_a_wrong_repo_cannot_poison_another(tmp_path):
    def handler(request):
        if request.url.path.startswith("/right/repo/"):
            return httpx.Response(200, text="code")
        return httpx.Response(404)

    wrong = GitHubRawCodeReader("wrong/repo", tmp_path, transport=httpx.MockTransport(handler))
    right = GitHubRawCodeReader("right/repo", tmp_path, transport=httpx.MockTransport(handler))
    assert asyncio.run(wrong.read(SHA, "a.ts")) is None
    assert asyncio.run(right.read(SHA, "a.ts")) == "code"
    for r in (wrong, right):
        asyncio.run(r.aclose())


def test_raw_reader_percent_encodes_the_path(tmp_path):
    calls: list[str] = []
    reader = _raw_reader(tmp_path, lambda r: httpx.Response(200, text="x"), calls)
    asyncio.run(reader.read(SHA, "src/odd#name?.ts"))
    asyncio.run(reader.aclose())
    assert calls[0].endswith("/src/odd%23name%3F.ts")


def test_local_reader_treats_a_directory_as_not_a_file(tmp_path):
    repo = tmp_path / "repo"
    (repo / "dir").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "dir" / "a.ts").write_text("x")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "one")
    head = _git(repo, "rev-parse", "HEAD")
    assert asyncio.run(LocalGitCodeReader(repo).read(head, "dir")) is None

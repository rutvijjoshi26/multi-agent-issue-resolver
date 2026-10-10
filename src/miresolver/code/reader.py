"""Read repository source as it stood at a pinned commit.

Investigators must see the code from before the fix, never the current
tree, or the fix itself leaks into the evidence they reason over.
"""

import asyncio
import os
import re
import tempfile
from urllib.parse import quote
from pathlib import Path, PurePosixPath
from typing import Protocol

import httpx

DEFAULT_RAW_URL = "https://raw.githubusercontent.com"
_COMMIT = re.compile(r"^[0-9a-fA-F]{7,40}$")
_MISSING_SUFFIX = ".__missing__"


class CodeReadError(Exception):
    """Raised when a file cannot be read for a reason other than absence."""


class CodeReader(Protocol):
    async def read(self, commit: str, path: str) -> str | None:
        """File text at the commit, or None when the file did not exist."""
        ...

    async def aclose(self) -> None: ...


def _check(commit: str, path: str) -> PurePosixPath:
    if not _COMMIT.match(commit or ""):
        raise CodeReadError(f"not a commit sha: {commit!r}")
    rel = PurePosixPath(path)
    if not path or rel.is_absolute() or ".." in rel.parts:
        raise CodeReadError(f"unsafe repository path: {path!r}")
    return rel


class GitHubRawCodeReader:
    """Fetches single files at a commit from GitHub, caching them on disk.

    An investigator needs a handful of files per case, so fetching them
    individually is far cheaper than cloning a large repository and
    checking out one commit per evaluation case. Absent files are cached
    too, so a repeat evaluation run makes no requests at all.
    """

    def __init__(
        self,
        repo: str,
        cache_dir: Path,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        base_url: str = DEFAULT_RAW_URL,
    ) -> None:
        self._repo = repo
        self._cache = Path(cache_dir)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            transport=transport,
            timeout=30.0,
            follow_redirects=True,
        )

    async def read(self, commit: str, path: str) -> str | None:
        rel = _check(commit, path)
        cached = self._cache / self._repo / commit / rel
        missing = cached.with_name(cached.name + _MISSING_SUFFIX)
        if cached.is_file():
            return cached.read_text(errors="replace")
        if missing.is_file():
            return None
        try:
            response = await self._client.get(
                f"/{self._repo}/{commit}/{quote(str(rel))}"
            )
        except httpx.HTTPError as exc:
            raise CodeReadError(f"fetching {path}@{commit} failed: {exc}") from exc
        cached.parent.mkdir(parents=True, exist_ok=True)
        if response.status_code == 404:
            missing.touch()
            return None
        if response.status_code != 200:
            raise CodeReadError(
                f"fetching {path}@{commit} returned HTTP {response.status_code}"
            )
        _atomic_write(cached, response.text)
        return response.text

    async def aclose(self) -> None:
        await self._client.aclose()


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temporary file so an interrupted run never leaves a
    truncated file in the cache to be served as the real content."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class LocalGitCodeReader:
    """Reads files at a commit from a local clone with `git show`."""

    def __init__(self, repo_path: Path) -> None:
        self._repo = Path(repo_path)

    async def _git(self, *args: str) -> tuple[int, bytes, str]:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=self._repo,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "LC_ALL": "C"},
        )
        out, err = await proc.communicate()
        return proc.returncode or 0, out, err.decode(errors="replace").strip()

    async def read(self, commit: str, path: str) -> str | None:
        rel = _check(commit, path)
        spec = f"{commit}:{rel}"
        code, out, err = await self._git("cat-file", "-t", spec)
        if code != 0:
            code, _, err = await self._git("cat-file", "-e", f"{commit}^{{commit}}")
            if code != 0:
                raise CodeReadError(f"commit {commit} not found in {self._repo}: {err}")
            return None
        if out.strip() != b"blob":
            return None
        code, out, err = await self._git("cat-file", "-p", spec)
        if code != 0:
            raise CodeReadError(f"git cat-file {spec} failed: {err}")
        return out.decode(errors="replace")

    async def aclose(self) -> None:
        return None

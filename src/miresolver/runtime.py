"""Builds the real clients behind the resolver from config.

Kept apart from the CLI so tests can replace the whole runtime with fakes
and still drive the real workflow graph.
"""

import contextlib
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

from qdrant_client import QdrantClient

from miresolver.agents.graph import ResolverDeps
from miresolver.agents.types import ResolverOptions
from miresolver.code.reader import CodeReader, GitHubRawCodeReader, LocalGitCodeReader
from miresolver.models.factory import build_chat, build_embedder
from miresolver.models.settings import ModelSettings
from miresolver.retrieval.index import IssueIndex
from miresolver.retrieval.pipeline import search

#: Seconds per Qdrant request. The client default of 5 turned brief server
#: stalls into failed evaluation cases.
QDRANT_TIMEOUT_S = 60


@contextlib.asynccontextmanager
async def open_runtime(
    settings: ModelSettings,
    env: Mapping[str, str],
    *,
    qdrant_url: str,
    collection: str,
    repo: str,
    local_repo: Path | None,
    code_cache: Path,
    options: ResolverOptions,
) -> AsyncIterator[ResolverDeps]:
    closers = []
    qdrant: QdrantClient | None = None
    try:
        embedder = build_embedder(settings, env)
        closers.append(embedder)
        chats = {}
        for tier in ("screen", "investigate", "synthesize"):
            chats[tier] = build_chat(settings, tier, env)
            closers.append(chats[tier])
        reader: CodeReader = (
            LocalGitCodeReader(local_repo)
            if local_repo is not None
            else GitHubRawCodeReader(repo, code_cache)
        )
        closers.append(reader)
        qdrant = QdrantClient(url=qdrant_url, timeout=QDRANT_TIMEOUT_S)
        index = IssueIndex(qdrant, collection=collection)

        async def retrieve(query: str, limit: int):
            return await search(
                query, embedder, index, limit=limit, mode=options.retrieval
            )

        yield ResolverDeps(
            retriever=retrieve,
            screen_chat=chats["screen"],
            investigate_chat=chats["investigate"],
            synthesize_chat=chats["synthesize"],
            reader=reader,
            options=options,
        )
    finally:
        for closer in closers:
            await closer.aclose()
        if qdrant is not None:
            qdrant.close()

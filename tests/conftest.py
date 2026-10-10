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


@pytest.fixture
def qdrant():
    """In-memory Qdrant, so index tests need no running service."""
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    yield client
    client.close()


@pytest.fixture
def live_qdrant():
    """Real Qdrant server, for behavior the in-memory client cannot exercise.

    In-memory mode makes payload text indexes a no-op and answers MatchText
    with an in-process scan, so it cannot catch a missing or misconfigured
    index. This fixture talks to a real server so that gap has coverage.
    """
    from qdrant_client import QdrantClient

    url = os.environ.get("TEST_QDRANT_URL")
    if not url:
        pytest.skip("TEST_QDRANT_URL not set")
    client = QdrantClient(url=url)
    yield client
    for collection in client.get_collections().collections:
        if collection.name.startswith("test_live_"):
            client.delete_collection(collection.name)
    client.close()

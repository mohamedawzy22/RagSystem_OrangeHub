import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("EMBEDDING_MODEL", "bge-m3")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_COLLECTION_NAME", "test_collection")
os.environ.setdefault("QDRANT_VECTOR_SIZE", "1024")
os.environ.setdefault("QDRANT_DISTANCE", "Cosine")


pytest_plugins = (
    "tests.fixtures.llm",
    "tests.fixtures.rag",
)


@pytest.fixture
def app(monkeypatch):
    from main import app as fastapi_app

    monkeypatch.setattr(
        fastapi_app,
        "db_client",
        MagicMock(),
        raising=False,
    )

    return fastapi_app


@pytest.fixture
def client(app):
    return TestClient(app)

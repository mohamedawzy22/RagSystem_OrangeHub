from unittest.mock import MagicMock

import pytest

from services.vectordb.providers.qdrant import QdrantVectorDB
from services.vectordb.vector_db_enum import DistanceMetric, VectorDBProvider
from services.vectordb.vectordb_factory import VectorDBFactory


def create_settings():
    settings = MagicMock()

    settings.QDRANT_URL = "http://localhost:6333"
    settings.QDRANT_API_KEY = None
    settings.QDRANT_COLLECTION_NAME = "test_collection"
    settings.QDRANT_VECTOR_SIZE = 1024
    settings.QDRANT_DISTANCE = DistanceMetric.COSINE

    return settings


def test_create_qdrant():
    settings = create_settings()

    factory = VectorDBFactory(settings)

    database = factory.create(VectorDBProvider.QDRANT)

    assert isinstance(database, QdrantVectorDB)
    assert database.collection_name == "test_collection"
    assert database.vector_size == 1024
    assert database.distance == DistanceMetric.COSINE


def test_create_qdrant_without_url():
    settings = create_settings()
    settings.QDRANT_URL = None

    factory = VectorDBFactory(settings)

    with pytest.raises(
        ValueError,
        match="QDRANT_URL is required",
    ):
        factory.create(VectorDBProvider.QDRANT)


def test_create_unsupported_provider():
    settings = create_settings()

    factory = VectorDBFactory(settings)

    with pytest.raises(
        ValueError,
        match="Unsupported vector database provider",
    ):
        factory.create("unknown")

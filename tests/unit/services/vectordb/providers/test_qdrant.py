from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http import models

from services.vectordb.providers.qdrant import QdrantVectorDB
from services.vectordb.vector_db_enum import DistanceMetric


def create_qdrant():
    with patch("services.vectordb.providers.qdrant.AsyncQdrantClient") as client_class:
        client = MagicMock()
        client_class.return_value = client

        database = QdrantVectorDB(
            url="http://localhost:6333",
            collection_name="test_collection",
            vector_size=1024,
            distance=DistanceMetric.COSINE,
        )

    return database, client


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (DistanceMetric.COSINE, models.Distance.COSINE),
        (DistanceMetric.EUCLID, models.Distance.EUCLID),
        (DistanceMetric.DOT, models.Distance.DOT),
    ],
)
def test_get_distance(distance, expected):
    with patch("services.vectordb.providers.qdrant.AsyncQdrantClient"):
        database = QdrantVectorDB(
            url="http://localhost:6333",
            collection_name="test_collection",
            vector_size=1024,
            distance=distance,
        )

    assert database._get_distance() == expected


@pytest.mark.anyio
async def test_create_collection_when_already_exists():
    database, client = create_qdrant()

    client.collection_exists = AsyncMock(
        return_value=True,
    )

    await database.create_collection()

    client.collection_exists.assert_awaited_once_with(
        collection_name="test_collection",
    )

    client.create_collection.assert_not_called()


@pytest.mark.anyio
async def test_create_collection_when_missing():
    database, client = create_qdrant()

    client.collection_exists = AsyncMock(
        return_value=False,
    )

    client.create_collection = AsyncMock()

    await database.create_collection()

    client.collection_exists.assert_awaited_once_with(
        collection_name="test_collection",
    )

    client.create_collection.assert_awaited_once()

    kwargs = client.create_collection.await_args.kwargs

    assert kwargs["collection_name"] == "test_collection"

    vector_config = kwargs["vectors_config"]

    assert isinstance(
        vector_config,
        models.VectorParams,
    )

    assert vector_config.size == 1024
    assert vector_config.distance == models.Distance.COSINE


@pytest.mark.anyio
async def test_delete_collection():
    database, client = create_qdrant()

    client.delete_collection = AsyncMock()

    await database.delete_collection()

    client.delete_collection.assert_awaited_once_with(
        collection_name="test_collection",
    )


@pytest.mark.anyio
async def test_upsert():
    database, client = create_qdrant()

    client.upsert = AsyncMock()

    vectors = [
        {
            "id": "point-1",
            "vector": [0.1, 0.2],
            "payload": {
                "text": "hello",
            },
        },
        {
            "id": "point-2",
            "vector": [0.3, 0.4],
        },
    ]

    await database.upsert(vectors)

    client.upsert.assert_awaited_once()

    kwargs = client.upsert.await_args.kwargs

    assert kwargs["collection_name"] == "test_collection"

    points = kwargs["points"]

    assert len(points) == 2

    assert isinstance(
        points[0],
        models.PointStruct,
    )

    assert points[0].id == "point-1"
    assert points[0].vector == [0.1, 0.2]
    assert points[0].payload == {"text": "hello"}

    assert points[1].id == "point-2"
    assert points[1].vector == [0.3, 0.4]
    assert points[1].payload == {}


@pytest.mark.anyio
async def test_search():
    database, client = create_qdrant()

    client.search = AsyncMock(
        return_value=[
            SimpleNamespace(
                id="point-1",
                score=0.95,
                payload={
                    "text": "hello",
                },
            ),
            SimpleNamespace(
                id="point-2",
                score=0.80,
                payload={
                    "text": "world",
                },
            ),
        ],
    )

    result = await database.search(
        vector=[0.1, 0.2],
        limit=5,
    )

    client.search.assert_awaited_once_with(
        collection_name="test_collection",
        query_vector=[0.1, 0.2],
        limit=5,
    )

    assert result == [
        {
            "id": "point-1",
            "score": 0.95,
            "payload": {
                "text": "hello",
            },
        },
        {
            "id": "point-2",
            "score": 0.80,
            "payload": {
                "text": "world",
            },
        },
    ]


@pytest.mark.anyio
async def test_search_without_payload():
    database, client = create_qdrant()

    client.search = AsyncMock(
        return_value=[
            SimpleNamespace(
                id="point-1",
                score=0.95,
                payload=None,
            ),
        ],
    )

    result = await database.search(
        vector=[0.1, 0.2],
    )

    assert result == [
        {
            "id": "point-1",
            "score": 0.95,
            "payload": None,
        }
    ]

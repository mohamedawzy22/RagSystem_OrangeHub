from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.rag.indexing_service import IndexingService


def test_build_qdrant_point_id_is_deterministic():
    first = IndexingService._build_qdrant_point_id("chunk-123")
    second = IndexingService._build_qdrant_point_id("chunk-123")

    assert first == second
    assert isinstance(first, str)


def test_build_qdrant_point_id_changes_for_different_chunks():
    first = IndexingService._build_qdrant_point_id("chunk-1")
    second = IndexingService._build_qdrant_point_id("chunk-2")

    assert first != second


@pytest.mark.anyio
async def test_index_embeds_chunks_and_upserts_vectors():
    embedding_model = AsyncMock()
    vector_db = AsyncMock()
    project_model = AsyncMock()
    chunk_model = AsyncMock()

    chunks = [
        SimpleNamespace(
            id="chunk-1",
            chunk_text="Lebanon is a country in the Middle East.",
            chunk_asset_id="asset-1",
        ),
        SimpleNamespace(
            id="chunk-2",
            chunk_text="Beirut is the capital of Lebanon.",
            chunk_asset_id="asset-1",
        ),
    ]

    project_model.get_project_or_create_one.return_value = SimpleNamespace(
        id="mongo-project-1"
    )

    async def chunk_iterator(
        *,
        project_id,
        batch_size,
    ):
        assert project_id == "mongo-project-1"
        assert batch_size == 2

        yield chunks

    chunk_model.iter_project_chunks = chunk_iterator

    embedding_model.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    service = IndexingService.__new__(IndexingService)

    service.embedding_model = embedding_model
    service.vector_db = vector_db
    service.project_model = project_model
    service.chunk_model = chunk_model

    result = await service.index(
        project_id="project1",
        batch_size=2,
    )

    project_model.get_project_or_create_one.assert_awaited_once_with(
        project_id="project1",
    )

    embedding_model.embed_documents.assert_awaited_once_with(
        [
            chunks[0].chunk_text,
            chunks[1].chunk_text,
        ]
    )

    vector_db.upsert.assert_awaited_once()

    vectors = vector_db.upsert.await_args.args[0]

    assert len(vectors) == 2

    assert vectors[0]["payload"] == {
        "chunk_id": "chunk-1",
        "project_id": "project1",
        "asset_id": "asset-1",
        "text": chunks[0].chunk_text,
    }

    assert vectors[1]["payload"] == {
        "chunk_id": "chunk-2",
        "project_id": "project1",
        "asset_id": "asset-1",
        "text": chunks[1].chunk_text,
    }

    assert vectors[0]["vector"] == [0.1, 0.2, 0.3]
    assert vectors[1]["vector"] == [0.4, 0.5, 0.6]

    assert result["project_id"] == "project1"
    assert result["assets"] == 1
    assert result["chunks"] == 2
    assert "message" in result

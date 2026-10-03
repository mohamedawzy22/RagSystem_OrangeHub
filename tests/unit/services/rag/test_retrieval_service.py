from unittest.mock import AsyncMock

import pytest

from services.rag.retrieval_service import RetrievalService


@pytest.mark.anyio
async def test_search_generates_embedding_and_queries_vector_db(
    sample_documents,
):
    embedding_model = AsyncMock()
    vector_db = AsyncMock()

    embedding_model.embed_text.return_value = [0.1, 0.2, 0.3]

    vector_db.search.return_value = [
        {
            "payload": {
                "text": document["text"],
            },
            "score": document["score"],
        }
        for document in sample_documents
    ]

    service = RetrievalService.__new__(RetrievalService)
    service.embedding_model = embedding_model
    service.vector_db = vector_db

    results = await service.search(
        project_id="project1",
        query="What is Lebanon?",
        limit=2,
    )

    embedding_model.embed_text.assert_awaited_once_with(
        "What is Lebanon?",
    )

    vector_db.search.assert_awaited_once_with(
        vector=[0.1, 0.2, 0.3],
        limit=2,
        project_id="project1",
    )

    assert results == sample_documents


@pytest.mark.anyio
async def test_search_skips_results_without_text():
    embedding_model = AsyncMock()
    vector_db = AsyncMock()

    embedding_model.embed_text.return_value = [0.1, 0.2]

    vector_db.search.return_value = [
        {
            "payload": {},
            "score": 0.9,
        },
        {
            "payload": {
                "text": "Valid chunk",
            },
            "score": 0.8,
        },
    ]

    service = RetrievalService.__new__(RetrievalService)
    service.embedding_model = embedding_model
    service.vector_db = vector_db

    results = await service.search(
        project_id="project1",
        query="test",
        limit=5,
    )

    assert results == [
        {
            "text": "Valid chunk",
            "score": 0.8,
        }
    ]

from unittest.mock import AsyncMock

import pytest

from services.rag.rag_service import RAGService


@pytest.mark.anyio
async def test_generate_uses_retrieved_documents(
    sample_documents,
):
    retrieval_service = AsyncMock()
    chat_model = AsyncMock()

    retrieval_service.search.return_value = sample_documents
    chat_model.generate.return_value = "Lebanon is a country in the Middle East."

    service = RAGService.__new__(RAGService)
    service.retrieval_service = retrieval_service
    service.chat_model = chat_model

    result = await service.generate(
        project_id="project1",
        query="What is Lebanon?",
        limit=2,
    )

    retrieval_service.search.assert_awaited_once_with(
        project_id="project1",
        query="What is Lebanon?",
        limit=2,
    )

    chat_model.generate.assert_awaited_once()

    prompt = chat_model.generate.await_args.kwargs["prompt"]

    assert "What is Lebanon?" in prompt
    assert sample_documents[0]["text"] in prompt
    assert sample_documents[1]["text"] in prompt

    assert result["query"] == "What is Lebanon?"
    assert result["answer"] == ("Lebanon is a country in the Middle East.")
    assert result["documents"] == sample_documents


@pytest.mark.anyio
async def test_generate_returns_no_information_when_retrieval_is_empty():
    retrieval_service = AsyncMock()
    chat_model = AsyncMock()

    retrieval_service.search.return_value = []

    service = RAGService.__new__(RAGService)
    service.retrieval_service = retrieval_service
    service.chat_model = chat_model

    result = await service.generate(
        project_id="project1",
        query="Unknown question",
        limit=5,
    )

    assert result["query"] == "Unknown question"
    assert result["documents"] == []
    assert "answer" in result

    chat_model.generate.assert_not_awaited()

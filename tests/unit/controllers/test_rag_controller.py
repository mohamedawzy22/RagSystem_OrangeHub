from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from controllers.rag_controller import RAGController
from models import RAGMessage
from models.chunk_model import ChunkModel
from models.project_model import ProjectModel
from tests.factories import make_chunk, make_project


def test_validate_project_id_invalid_type():
    with pytest.raises(
        TypeError,
        match="Project ID must be a string",
    ):
        RAGController._validate_project_id(123)


@pytest.mark.parametrize("project_id", ["", "   "])
def test_validate_project_id_empty(project_id):
    with pytest.raises(
        ValueError,
        match="Project ID cannot be empty",
    ):
        RAGController._validate_project_id(project_id)


def test_validate_query_invalid_type():
    with pytest.raises(
        TypeError,
        match="Query must be a string",
    ):
        RAGController._validate_query(123)


@pytest.mark.parametrize("query", ["", "   "])
def test_validate_query_empty(query):
    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        RAGController._validate_query(query)


def test_validate_limit_invalid_type():
    with pytest.raises(
        TypeError,
        match="Limit must be an integer",
    ):
        RAGController._validate_limit("5")


@pytest.mark.parametrize("limit", [0, -1])
def test_validate_limit_invalid_value(limit):
    with pytest.raises(
        ValueError,
        match="Limit must be greater than zero",
    ):
        RAGController._validate_limit(limit)


def test_build_qdrant_point_id_is_deterministic():
    first = RAGController._build_qdrant_point_id("chunk-123")
    second = RAGController._build_qdrant_point_id("chunk-123")

    assert first == second
    assert isinstance(first, str)


def test_build_prompt_with_documents():
    results = [
        {
            "text": "RAG retrieves relevant documents.",
            "score": 0.95,
        },
        {
            "text": "The LLM generates the final response.",
            "score": 0.88,
        },
    ]

    prompt, documents = RAGController._build_prompt(
        query="What is RAG?",
        results=results,
    )

    assert "What is RAG?" in prompt
    assert "RAG retrieves relevant documents." in prompt
    assert "The LLM generates the final response." in prompt
    assert documents == results


def test_build_prompt_without_documents():
    prompt, documents = RAGController._build_prompt(
        query="What is RAG?",
        results=[],
    )

    assert "What is RAG?" in prompt
    assert documents == []


@pytest.mark.anyio
async def test_search_success(
    rag_controller,
    mock_embedding_model,
    mock_vector_db,
):
    mock_embedding_model.embed_text.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    mock_vector_db.search.return_value = [
        {
            "score": 0.95,
            "payload": {
                "text": "RAG retrieves relevant documents.",
            },
        },
        {
            "score": 0.82,
            "payload": {
                "text": "Embeddings represent semantic meaning.",
            },
        },
    ]

    results = await rag_controller.search(
        query="What is RAG?",
        limit=5,
    )

    assert results == [
        {
            "text": "RAG retrieves relevant documents.",
            "score": 0.95,
        },
        {
            "text": "Embeddings represent semantic meaning.",
            "score": 0.82,
        },
    ]

    mock_embedding_model.embed_text.assert_awaited_once_with(
        "What is RAG?",
    )

    mock_vector_db.search.assert_awaited_once_with(
        vector=[0.1, 0.2, 0.3],
        limit=5,
    )


@pytest.mark.anyio
async def test_search_skips_result_without_text(
    rag_controller,
    mock_embedding_model,
    mock_vector_db,
):
    mock_embedding_model.embed_text.return_value = [
        0.1,
        0.2,
    ]

    mock_vector_db.search.return_value = [
        {
            "score": 0.95,
            "payload": {
                "text": "Valid result",
            },
        },
        {
            "score": 0.50,
            "payload": {},
        },
    ]

    results = await rag_controller.search(
        query="test",
        limit=5,
    )

    assert results == [
        {
            "text": "Valid result",
            "score": 0.95,
        }
    ]


@pytest.mark.anyio
async def test_search_propagates_embedding_error(
    rag_controller,
    mock_embedding_model,
):
    mock_embedding_model.embed_text.side_effect = RuntimeError("Embedding failed")

    with pytest.raises(
        RuntimeError,
        match="Embedding failed",
    ):
        await rag_controller.search(
            query="test",
            limit=5,
        )


@pytest.mark.anyio
async def test_generate_success(
    rag_controller,
    mock_chat_model,
):
    rag_controller.search = AsyncMock(
        return_value=[
            {
                "text": "RAG retrieves relevant documents.",
                "score": 0.95,
            }
        ],
    )

    mock_chat_model.generate.return_value = (
        "RAG retrieves documents and generates an answer."
    )

    result = await rag_controller.generate(
        query="What is RAG?",
        limit=5,
    )

    assert result["query"] == "What is RAG?"

    assert result["answer"] == "RAG retrieves documents and generates an answer."

    assert result["documents"] == [
        {
            "text": "RAG retrieves relevant documents.",
            "score": 0.95,
        }
    ]

    rag_controller.search.assert_awaited_once_with(
        query="What is RAG?",
        limit=5,
    )

    mock_chat_model.generate.assert_awaited_once()


@pytest.mark.anyio
async def test_generate_without_relevant_documents(
    rag_controller,
    mock_chat_model,
):
    rag_controller.search = AsyncMock(
        return_value=[],
    )

    result = await rag_controller.generate(
        query="What is RAG?",
        limit=5,
    )

    assert result["query"] == "What is RAG?"
    assert result["documents"] == []

    assert result["answer"] == RAGMessage.NO_RELEVANT_INFORMATION.value

    assert "What is RAG?" in result["full_prompt"]

    mock_chat_model.generate.assert_not_called()


@pytest.mark.anyio
async def test_generate_propagates_chat_error(
    rag_controller,
    mock_chat_model,
):
    rag_controller.search = AsyncMock(
        return_value=[
            {
                "text": "RAG information",
                "score": 0.9,
            }
        ],
    )

    mock_chat_model.generate.side_effect = RuntimeError("Generation failed")

    with pytest.raises(
        RuntimeError,
        match="Generation failed",
    ):
        await rag_controller.generate(
            query="What is RAG?",
            limit=5,
        )


@pytest.mark.anyio
async def test_index_without_chunks(
    rag_controller,
    mock_embedding_model,
    mock_vector_db,
    monkeypatch,
):
    project = make_project()

    project_model = SimpleNamespace(
        get_project_or_create_one=AsyncMock(
            return_value=project,
        )
    )

    chunk_model = SimpleNamespace(
        get_project_chunks=AsyncMock(
            return_value=[],
        )
    )

    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        AsyncMock(return_value=project_model),
    )

    monkeypatch.setattr(
        ChunkModel,
        "create_instance",
        AsyncMock(return_value=chunk_model),
    )

    result = await rag_controller.index(
        project_id="project-1",
    )

    assert result == {
        "message": RAGMessage.NO_CHUNKS.value,
        "project_id": "project-1",
        "assets": 0,
        "chunks": 0,
    }

    mock_embedding_model.embed_documents.assert_not_called()
    mock_vector_db.upsert.assert_not_called()


@pytest.mark.anyio
async def test_index_success(
    rag_controller,
    mock_embedding_model,
    mock_vector_db,
    monkeypatch,
):
    project = make_project()

    chunks = [
        make_chunk(
            object_id="chunk-1",
            asset_id="asset-1",
            text="First chunk",
        ),
        make_chunk(
            object_id="chunk-2",
            asset_id="asset-2",
            text="Second chunk",
        ),
    ]

    project_model = SimpleNamespace(
        get_project_or_create_one=AsyncMock(
            return_value=project,
        )
    )

    chunk_model = SimpleNamespace(
        get_project_chunks=AsyncMock(
            return_value=chunks,
        )
    )

    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        AsyncMock(return_value=project_model),
    )

    monkeypatch.setattr(
        ChunkModel,
        "create_instance",
        AsyncMock(return_value=chunk_model),
    )

    mock_embedding_model.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    result = await rag_controller.index(
        project_id="project-1",
    )

    assert result["message"] == RAGMessage.INDEX_SUCCESS.value
    assert result["project_id"] == "project-1"
    assert result["assets"] == 2
    assert result["chunks"] == 2

    mock_embedding_model.embed_documents.assert_awaited_once_with(
        [
            "First chunk",
            "Second chunk",
        ]
    )

    mock_vector_db.upsert.assert_awaited_once()

    vectors = mock_vector_db.upsert.await_args.args[0]

    assert len(vectors) == 2
    assert vectors[0]["vector"] == [0.1, 0.2, 0.3]
    assert vectors[0]["payload"]["chunk_id"] == "chunk-1"
    assert vectors[0]["payload"]["project_id"] == "project-1"
    assert vectors[0]["payload"]["asset_id"] == "asset-1"
    assert vectors[0]["payload"]["text"] == "First chunk"


@pytest.mark.anyio
async def test_index_embedding_count_mismatch(
    rag_controller,
    mock_embedding_model,
    mock_vector_db,
    monkeypatch,
):
    project = make_project()

    chunks = [
        make_chunk(
            object_id="chunk-1",
            asset_id="asset-1",
            text="First chunk",
        ),
        make_chunk(
            object_id="chunk-2",
            asset_id="asset-2",
            text="Second chunk",
        ),
    ]

    project_model = SimpleNamespace(
        get_project_or_create_one=AsyncMock(
            return_value=project,
        )
    )

    chunk_model = SimpleNamespace(
        get_project_chunks=AsyncMock(
            return_value=chunks,
        )
    )

    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        AsyncMock(return_value=project_model),
    )

    monkeypatch.setattr(
        ChunkModel,
        "create_instance",
        AsyncMock(return_value=chunk_model),
    )

    mock_embedding_model.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
    ]

    with pytest.raises(
        ValueError,
        match="Number of embeddings does not match number of chunks",
    ):
        await rag_controller.index(
            project_id="project-1",
        )

    mock_vector_db.upsert.assert_not_called()


@pytest.mark.anyio
async def test_index_inconsistent_embedding_dimensions(
    rag_controller,
    mock_embedding_model,
    mock_vector_db,
    monkeypatch,
):
    project = make_project()

    chunks = [
        make_chunk(
            object_id="chunk-1",
            asset_id="asset-1",
            text="First chunk",
        ),
        make_chunk(
            object_id="chunk-2",
            asset_id="asset-2",
            text="Second chunk",
        ),
    ]

    project_model = SimpleNamespace(
        get_project_or_create_one=AsyncMock(
            return_value=project,
        )
    )

    chunk_model = SimpleNamespace(
        get_project_or_create_one=AsyncMock(
            return_value=project,
        )
    )

    chunk_model.get_project_chunks = AsyncMock(
        return_value=chunks,
    )

    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        AsyncMock(return_value=project_model),
    )

    monkeypatch.setattr(
        ChunkModel,
        "create_instance",
        AsyncMock(return_value=chunk_model),
    )

    mock_embedding_model.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5],
    ]

    with pytest.raises(
        ValueError,
        match="Embedding dimensions are inconsistent",
    ):
        await rag_controller.index(
            project_id="project-1",
        )

    mock_vector_db.upsert.assert_not_called()

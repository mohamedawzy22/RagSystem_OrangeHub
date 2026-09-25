from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_generate_success():
    mock_controller = MagicMock()
    mock_controller.generate = AsyncMock(
        return_value={
            "query": "Explain RAG",
            "answer": "RAG retrieves relevant information before generating an answer.",
            "full_prompt": "RAG prompt",
            "documents": [
                {
                    "text": "RAG combines retrieval with generation.",
                    "score": 0.95,
                }
            ],
        }
    )

    app.rag_controller = mock_controller

    response = client.post(
        "/api/v1/rag/generate",
        json={
            "query": "Explain RAG",
            "limit": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "Explain RAG"
    assert "RAG" in data["answer"]
    assert data["full_prompt"] == "RAG prompt"
    assert len(data["documents"]) == 1

    mock_controller.generate.assert_awaited_once_with(
        query="Explain RAG",
        limit=5,
    )


def test_generate_empty_query():
    response = client.post(
        "/api/v1/rag/generate",
        json={
            "query": "",
        },
    )

    assert response.status_code == 422


def test_generate_invalid_limit():
    response = client.post(
        "/api/v1/rag/generate",
        json={
            "query": "Hello",
            "limit": 0,
        },
    )

    assert response.status_code == 422


def test_generate_value_error():
    mock_controller = MagicMock()
    mock_controller.generate = AsyncMock(
        side_effect=ValueError("Query cannot be empty")
    )

    app.rag_controller = mock_controller

    response = client.post(
        "/api/v1/rag/generate",
        json={
            "query": "Hello",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty"


def test_generate_internal_error():
    mock_controller = MagicMock()
    mock_controller.generate = AsyncMock(side_effect=RuntimeError("Generation failed"))

    app.rag_controller = mock_controller

    response = client.post(
        "/api/v1/rag/generate",
        json={
            "query": "Hello",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to generate response"


def test_search_success():
    mock_controller = MagicMock()
    mock_controller.search = AsyncMock(
        return_value=[
            {
                "text": "RAG is a retrieval augmented generation system.",
                "score": 0.91,
            }
        ]
    )

    app.rag_controller = mock_controller

    response = client.post(
        "/api/v1/rag/search",
        json={
            "query": "What is RAG?",
            "limit": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "What is RAG?"
    assert len(data["results"]) == 1
    assert data["results"][0]["score"] == 0.91

    mock_controller.search.assert_awaited_once_with(
        query="What is RAG?",
        limit=5,
    )


def test_search_value_error():
    mock_controller = MagicMock()
    mock_controller.search = AsyncMock(side_effect=ValueError("Query cannot be empty"))

    app.rag_controller = mock_controller

    response = client.post(
        "/api/v1/rag/search",
        json={
            "query": "Hello",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty"


def test_search_internal_error():
    mock_controller = MagicMock()
    mock_controller.search = AsyncMock(side_effect=RuntimeError("Search failed"))

    app.rag_controller = mock_controller

    response = client.post(
        "/api/v1/rag/search",
        json={
            "query": "Hello",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to search documents"


def test_index_success():
    mock_controller = MagicMock()
    mock_controller.index = AsyncMock(
        return_value={
            "message": "Project indexed successfully",
            "project_id": "project-1",
            "assets": 2,
            "chunks": 10,
        }
    )

    app.rag_controller = mock_controller

    response = client.post(
        "/api/v1/rag/index/project-1",
    )

    assert response.status_code == 200

    data = response.json()

    assert data["project_id"] == "project-1"
    assert data["assets"] == 2
    assert data["chunks"] == 10

    mock_controller.index.assert_awaited_once_with(
        project_id="project-1",
    )


def test_index_value_error():
    mock_controller = MagicMock()
    mock_controller.index = AsyncMock(
        side_effect=ValueError("Project ID cannot be empty")
    )

    app.rag_controller = mock_controller

    response = client.post(
        "/api/v1/rag/index/project-1",
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Project ID cannot be empty"


def test_index_internal_error():
    mock_controller = MagicMock()
    mock_controller.index = AsyncMock(side_effect=RuntimeError("Indexing failed"))

    app.rag_controller = mock_controller

    response = client.post(
        "/api/v1/rag/index/project-1",
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to index project"

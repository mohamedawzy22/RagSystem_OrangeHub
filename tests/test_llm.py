from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from main import app
from services.llm.llm_enum import ChatModelName

client = TestClient(app)


def test_chat_generate_success(monkeypatch):
    mock_model = MagicMock()
    mock_model.generate = AsyncMock(
        return_value="RAG is a system that retrieves relevant information before generating an answer."
    )

    mock_manager = MagicMock()
    mock_manager.get_chat_model.return_value = mock_model

    app.model_manager = mock_manager

    response = client.post(
        "/api/v1/chat/generate",
        json={
            "model": ChatModelName.QWEN3.value,
            "prompt": "Explain RAG",
            "temperature": 0.7,
            "max_tokens": 100,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == ChatModelName.QWEN3.value
    assert "RAG" in data["response"]

    mock_manager.get_chat_model.assert_called_once_with(ChatModelName.QWEN3.value)

    mock_model.generate.assert_awaited_once_with(
        prompt="Explain RAG",
        temperature=0.7,
        max_tokens=100,
    )


def test_chat_generate_invalid_model():
    response = client.post(
        "/api/v1/chat/generate",
        json={
            "model": "invalid_model",
            "prompt": "Hello",
        },
    )

    assert response.status_code == 422


def test_chat_generate_empty_prompt():
    response = client.post(
        "/api/v1/chat/generate",
        json={
            "model": ChatModelName.QWEN3.value,
            "prompt": "",
        },
    )

    assert response.status_code == 422


def test_chat_generate_model_not_configured(monkeypatch):
    mock_manager = MagicMock()

    mock_manager.get_chat_model.side_effect = ValueError("Chat model is not configured")

    app.model_manager = mock_manager

    response = client.post(
        "/api/v1/chat/generate",
        json={
            "model": ChatModelName.QWEN3.value,
            "prompt": "Hello",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Chat model is not configured"


def test_chat_generate_internal_error(monkeypatch):
    mock_model = MagicMock()

    mock_model.generate = AsyncMock(side_effect=RuntimeError("Generation failed"))

    mock_manager = MagicMock()
    mock_manager.get_chat_model.return_value = mock_model

    app.model_manager = mock_manager

    response = client.post(
        "/api/v1/chat/generate",
        json={
            "model": ChatModelName.QWEN3.value,
            "prompt": "Hello",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to generate response"

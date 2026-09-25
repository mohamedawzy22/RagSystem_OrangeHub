from types import SimpleNamespace
from unittest.mock import AsyncMock


def test_rag_generate_e2e(client, app, monkeypatch):
    mock_controller = SimpleNamespace(
        generate=AsyncMock(
            return_value={
                "query": "What is Python?",
                "answer": "Python is a programming language.",
                "full_prompt": "Context...",
                "documents": [
                    {
                        "text": "Python is a programming language.",
                        "score": 0.95,
                    }
                ],
            }
        )
    )

    monkeypatch.setattr(
        app,
        "rag_controller",
        mock_controller,
        raising=False,
    )

    response = client.post(
        "/api/v1/rag/generate",
        json={
            "query": "What is Python?",
            "limit": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "What is Python?"
    assert "Python is a programming language" in data["answer"]
    assert len(data["documents"]) == 1

    mock_controller.generate.assert_awaited_once_with(
        query="What is Python?",
        limit=3,
    )

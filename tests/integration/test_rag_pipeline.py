import asyncio
from unittest.mock import MagicMock

import pytest

from controllers.rag_controller import RAGController


class FakeEmbeddingModel:
    async def embed_text(self, text):
        assert text
        return [0.1, 0.2, 0.3]


class FakeVectorDB:
    async def search(self, vector, limit=5):
        assert vector == [0.1, 0.2, 0.3]
        assert limit == 3

        return [
            {
                "id": "doc-1",
                "score": 0.95,
                "payload": {
                    "text": "Python is a programming language.",
                },
            },
            {
                "id": "doc-2",
                "score": 0.90,
                "payload": {
                    "text": "FastAPI is a Python web framework.",
                },
            },
        ]


class FakeVectorDBManager:
    def __init__(self):
        self.database = FakeVectorDB()

    def get_database(self, name):
        assert name == "default"
        return self.database


class FakeChatModel:
    async def generate(self, prompt):
        assert "Python" in prompt
        assert "FastAPI" in prompt
        return (
            "Python is a programming language, and FastAPI is a Python web framework."
        )


@pytest.fixture
def integration_rag_controller():
    return RAGController(
        chat_model=FakeChatModel(),
        embedding_model=FakeEmbeddingModel(),
        vector_db_manager=FakeVectorDBManager(),
        db_client=MagicMock(),
    )


def test_rag_search_pipeline(integration_rag_controller):
    result = asyncio.run(
        integration_rag_controller.search(
            query="What is Python?",
            limit=3,
        )
    )

    assert len(result) == 2
    assert result[0]["text"] == "Python is a programming language."
    assert result[0]["score"] == 0.95
    assert result[1]["text"] == "FastAPI is a Python web framework."


def test_rag_generate_pipeline(integration_rag_controller):
    result = asyncio.run(
        integration_rag_controller.generate(
            query="What are Python and FastAPI?",
            limit=3,
        )
    )

    assert result["query"] == "What are Python and FastAPI?"
    assert "Python is a programming language" in result["answer"]
    assert "FastAPI is a Python web framework" in result["answer"]

    assert result["full_prompt"]
    assert "Python is a programming language." in result["full_prompt"]
    assert "FastAPI is a Python web framework." in result["full_prompt"]

    assert len(result["documents"]) == 2
    assert result["documents"][0]["text"] == "Python is a programming language."

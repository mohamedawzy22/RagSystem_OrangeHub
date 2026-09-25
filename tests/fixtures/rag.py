from unittest.mock import AsyncMock, MagicMock

import pytest

from controllers.rag_controller import RAGController


@pytest.fixture
def mock_chat_model():
    model = MagicMock()
    model.generate = AsyncMock()

    return model


@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    model.embed_text = AsyncMock()
    model.embed_documents = AsyncMock()

    return model


@pytest.fixture
def mock_vector_db():
    database = MagicMock()
    database.search = AsyncMock()
    database.upsert = AsyncMock()

    return database


@pytest.fixture
def rag_controller(
    mock_chat_model,
    mock_embedding_model,
    mock_vector_db,
):
    vector_db_manager = MagicMock()

    vector_db_manager.get_database.return_value = mock_vector_db

    return RAGController(
        chat_model=mock_chat_model,
        embedding_model=mock_embedding_model,
        vector_db_manager=vector_db_manager,
        db_client=MagicMock(),
    )

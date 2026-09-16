from unittest.mock import MagicMock

import pytest

from helpers.config import ChatModelSettings, EmbeddingModelSettings
from services.llm.factory import ModelFactory
from services.llm.providers.ollama import (
    OllamaChatModel,
    OllamaEmbeddingModel,
)
from services.llm.providers.openrouter import (
    OpenRouterChatModel,
    OpenRouterEmbeddingModel,
)


def create_settings(
    ollama_url="http://localhost:11434",
    openrouter_key="test-key",
):
    settings = MagicMock()

    settings.OLLAMA_BASE_URL = ollama_url
    settings.OPENROUTER_API_KEY = openrouter_key

    return settings


def test_create_ollama_chat_model():
    settings = create_settings()
    factory = ModelFactory(settings)

    config = ChatModelSettings(
        provider="ollama",
        model="qwen3:8b",
    )

    model = factory.create_chat_model(config)

    assert isinstance(model, OllamaChatModel)
    assert model.model_id == "qwen3:8b"


def test_create_openrouter_chat_model():
    settings = create_settings()
    factory = ModelFactory(settings)

    config = ChatModelSettings(
        provider="openrouter",
        model="test-model",
    )

    model = factory.create_chat_model(config)

    assert isinstance(model, OpenRouterChatModel)
    assert model.model_id == "test-model"


def test_create_ollama_chat_model_without_url():
    settings = create_settings(ollama_url=None)
    factory = ModelFactory(settings)

    config = ChatModelSettings(
        provider="ollama",
        model="qwen3:8b",
    )

    with pytest.raises(
        ValueError,
        match="OLLAMA_BASE_URL is required",
    ):
        factory.create_chat_model(config)


def test_create_openrouter_chat_model_without_key():
    settings = create_settings(openrouter_key=None)
    factory = ModelFactory(settings)

    config = ChatModelSettings(
        provider="openrouter",
        model="test-model",
    )

    with pytest.raises(
        ValueError,
        match="OPENROUTER_API_KEY is required",
    ):
        factory.create_chat_model(config)


def test_create_chat_model_unsupported_provider():
    settings = create_settings()
    factory = ModelFactory(settings)

    config = ChatModelSettings(
        provider="unknown",
        model="test-model",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported chat provider: unknown",
    ):
        factory.create_chat_model(config)


def test_create_ollama_embedding_model():
    settings = create_settings()
    factory = ModelFactory(settings)

    config = EmbeddingModelSettings(
        provider="ollama",
        model="bge-m3",
        dimension=1024,
    )

    model = factory.create_embedding_model(config)

    assert isinstance(model, OllamaEmbeddingModel)
    assert model.model_id == "bge-m3"
    assert model.dimension == 1024


def test_create_openrouter_embedding_model():
    settings = create_settings()
    factory = ModelFactory(settings)

    config = EmbeddingModelSettings(
        provider="openrouter",
        model="test-embedding-model",
        dimension=1024,
    )

    model = factory.create_embedding_model(config)

    assert isinstance(model, OpenRouterEmbeddingModel)
    assert model.model_id == "test-embedding-model"
    assert model.dimension == 1024


def test_create_ollama_embedding_model_without_url():
    settings = create_settings(ollama_url=None)
    factory = ModelFactory(settings)

    config = EmbeddingModelSettings(
        provider="ollama",
        model="bge-m3",
        dimension=1024,
    )

    with pytest.raises(
        ValueError,
        match="OLLAMA_BASE_URL is required",
    ):
        factory.create_embedding_model(config)


def test_create_openrouter_embedding_model_without_key():
    settings = create_settings(openrouter_key=None)
    factory = ModelFactory(settings)

    config = EmbeddingModelSettings(
        provider="openrouter",
        model="test-model",
        dimension=1024,
    )

    with pytest.raises(
        ValueError,
        match="OPENROUTER_API_KEY is required",
    ):
        factory.create_embedding_model(config)


def test_create_embedding_model_unsupported_provider():
    settings = create_settings()
    factory = ModelFactory(settings)

    config = EmbeddingModelSettings(
        provider="unknown",
        model="test-model",
        dimension=1024,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported embedding provider: unknown",
    ):
        factory.create_embedding_model(config)

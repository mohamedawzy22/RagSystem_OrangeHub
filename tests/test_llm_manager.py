from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.llm.fallback_chat import FallbackChatModel
from services.llm.llm_manager import ModelManager


def test_load_models():
    chat_model = MagicMock()
    fallback_chat_model = MagicMock()
    embedding_model = MagicMock()

    factory = MagicMock()

    factory.settings.CHAT_MODELS = {
        "qwen3": SimpleNamespace(
            provider="ollama",
            model="qwen3:8b",
        ),
        "fallback": SimpleNamespace(
            provider="ollama",
            model="llama3.2:3b",
        ),
    }

    factory.settings.EMBEDDING_MODELS = {
        "bge-m3": SimpleNamespace(
            provider="ollama",
            model="bge-m3",
            dimension=1024,
        ),
    }

    factory.settings.CHAT_PRIMARY_MODEL = "qwen3"
    factory.settings.CHAT_FALLBACK_MODEL = "fallback"
    factory.settings.EMBEDDING_MODEL = "bge-m3"

    factory.create_chat_model.side_effect = [
        chat_model,
        fallback_chat_model,
    ]

    factory.create_embedding_model.return_value = embedding_model

    manager = ModelManager(factory)

    manager.load_models()

    assert manager._chat_models["qwen3"] is chat_model
    assert manager._chat_models["fallback"] is fallback_chat_model

    assert manager._embedding_models["bge-m3"] is embedding_model

    factory.create_chat_model.assert_called()
    factory.create_embedding_model.assert_called_once()


def test_get_chat_model_success():
    factory = MagicMock()
    manager = ModelManager(factory)

    chat_model = MagicMock()
    manager._chat_models["qwen3"] = chat_model

    result = manager.get_chat_model("qwen3")

    assert result is chat_model


def test_get_chat_model_not_configured():
    factory = MagicMock()
    manager = ModelManager(factory)

    with pytest.raises(
        ValueError,
        match="Chat model 'qwen3' is not configured",
    ):
        manager.get_chat_model("qwen3")


def test_get_embedding_model_success():
    factory = MagicMock()
    manager = ModelManager(factory)

    embedding_model = MagicMock()
    manager._embedding_models["bge-m3"] = embedding_model

    result = manager.get_embedding_model("bge-m3")

    assert result is embedding_model


def test_get_embedding_model_not_configured():
    factory = MagicMock()
    manager = ModelManager(factory)

    with pytest.raises(
        ValueError,
        match="Embedding model 'bge-m3' is not configured",
    ):
        manager.get_embedding_model("bge-m3")


def test_selected_models_are_initialized():
    chat_model = MagicMock()
    fallback_chat_model = MagicMock()
    embedding_model = MagicMock()

    factory = MagicMock()

    factory.settings.CHAT_MODELS = {
        "qwen3": SimpleNamespace(
            provider="ollama",
            model="qwen3:8b",
        ),
        "fallback": SimpleNamespace(
            provider="ollama",
            model="llama3.2:3b",
        ),
    }

    factory.settings.EMBEDDING_MODELS = {
        "bge-m3": SimpleNamespace(
            provider="ollama",
            model="bge-m3",
            dimension=1024,
        ),
    }

    factory.settings.CHAT_PRIMARY_MODEL = "qwen3"
    factory.settings.CHAT_FALLBACK_MODEL = "fallback"
    factory.settings.EMBEDDING_MODEL = "bge-m3"

    factory.create_chat_model.side_effect = [
        chat_model,
        fallback_chat_model,
    ]

    factory.create_embedding_model.return_value = embedding_model

    manager = ModelManager(factory)

    manager.load_models()

    selected_chat = manager.get_selected_chat_model()
    selected_embedding = manager.get_selected_embedding_model()

    assert isinstance(selected_chat, FallbackChatModel)
    assert selected_chat.primary is chat_model
    assert selected_chat.fallback is fallback_chat_model

    assert selected_embedding is embedding_model


def test_selected_chat_model_not_initialized():
    factory = MagicMock()
    manager = ModelManager(factory)

    with pytest.raises(
        RuntimeError,
        match="Selected chat model is not initialized",
    ):
        manager.get_selected_chat_model()


def test_selected_embedding_model_not_initialized():
    factory = MagicMock()
    manager = ModelManager(factory)

    with pytest.raises(
        RuntimeError,
        match="Selected embedding model is not initialized",
    ):
        manager.get_selected_embedding_model()

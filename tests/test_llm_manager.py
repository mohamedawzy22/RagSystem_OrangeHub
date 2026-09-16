from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.llm.manager import ModelManager


def test_load_models():
    chat_model = MagicMock()
    embedding_model = MagicMock()

    factory = MagicMock()

    factory.settings.CHAT_MODELS = {
        "qwen3": SimpleNamespace(provider="ollama", model="qwen3:8b"),
    }

    factory.settings.EMBEDDING_MODELS = {
        "bge-m3": SimpleNamespace(
            provider="ollama",
            model="bge-m3",
            dimension=1024,
        ),
    }

    factory.create_chat_model.return_value = chat_model
    factory.create_embedding_model.return_value = embedding_model

    manager = ModelManager(factory)

    manager.load_models()

    assert manager._chat_models["qwen3"] is chat_model
    assert manager._embedding_models["bge-m3"] is embedding_model

    factory.create_chat_model.assert_called_once()
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

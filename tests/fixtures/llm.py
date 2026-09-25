from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_model_factory():
    factory = MagicMock()

    factory.settings.CHAT_MODELS = {
        "qwen3": SimpleNamespace(
            provider="ollama",
            model="qwen2.5:3b",
        ),
        "fallback": SimpleNamespace(
            provider="ollama",
            model="llama3.2:3b",
        ),
    }

    factory.settings.EMBEDDING_MODELS = {
        "bge-m3": SimpleNamespace(
            provider="ollama",
            model="bge-m3:latest",
            dimension=1024,
        ),
    }

    factory.settings.CHAT_PRIMARY_MODEL = "qwen3"
    factory.settings.CHAT_FALLBACK_MODEL = "fallback"
    factory.settings.EMBEDDING_MODEL = "bge-m3"

    return factory

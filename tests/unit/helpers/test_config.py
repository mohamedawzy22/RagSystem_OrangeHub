import pytest

from helpers.config import (
    ChatModelSettings,
    EmbeddingModelSettings,
    Setting,
)
from services.vectordb.vector_db_enum import DistanceMetric


def valid_settings(**overrides):
    data = {
        "APP_NAME": "test",
        "APP_VERSION": "1.0",
        "FILE_ALLOWED_TYPES": ["text/plain"],
        "FILE_MAX_SIZE": 10,
        "FILE_DEFAULT_CHUNK_SIZE": 100,
        "MONGODB_URL": "mongodb://localhost",
        "MONGODB_DATABASE": "test-db",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "CHAT_MODELS": {
            "primary": ChatModelSettings(
                provider="ollama",
                model="qwen3:8b",
            ),
            "fallback": ChatModelSettings(
                provider="ollama",
                model="llama3.2:3b",
            ),
        },
        "EMBEDDING_MODELS": {
            "bge-m3": EmbeddingModelSettings(
                provider="ollama",
                model="bge-m3",
                dimension=1024,
            ),
        },
        "CHAT_PRIMARY_MODEL": "primary",
        "CHAT_FALLBACK_MODEL": "fallback",
        "CHAT_FALLBACK_COOLDOWN_SECONDS": 60,
        "EMBEDDING_MODEL": "bge-m3",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_COLLECTION_NAME": "test",
        "QDRANT_VECTOR_SIZE": 1024,
        "QDRANT_DISTANCE": DistanceMetric.COSINE,
    }

    data.update(overrides)
    return data


def test_valid_model_configuration():
    settings = Setting(**valid_settings())

    assert settings.CHAT_PRIMARY_MODEL == "primary"
    assert settings.CHAT_FALLBACK_MODEL == "fallback"
    assert settings.EMBEDDING_MODEL == "bge-m3"


def test_invalid_chat_primary_model():
    with pytest.raises(ValueError, match="primary model"):
        Setting(
            **valid_settings(
                CHAT_PRIMARY_MODEL="missing",
            )
        )


def test_invalid_chat_fallback_model():
    with pytest.raises(ValueError, match="fallback model"):
        Setting(
            **valid_settings(
                CHAT_FALLBACK_MODEL="missing",
            )
        )


def test_chat_primary_and_fallback_must_differ():
    with pytest.raises(ValueError, match="must be different"):
        Setting(
            **valid_settings(
                CHAT_FALLBACK_MODEL="primary",
            )
        )


def test_invalid_embedding_model():
    with pytest.raises(ValueError, match="Embedding model"):
        Setting(
            **valid_settings(
                EMBEDDING_MODEL="missing",
            )
        )


def test_negative_fallback_cooldown():
    with pytest.raises(ValueError, match="cannot be negative"):
        Setting(
            **valid_settings(
                CHAT_FALLBACK_COOLDOWN_SECONDS=-1,
            )
        )


def test_invalid_qdrant_vector_size():
    with pytest.raises(ValueError, match="QDRANT_VECTOR_SIZE"):
        Setting(
            **valid_settings(
                QDRANT_VECTOR_SIZE=0,
            )
        )


def test_embedding_dimension_must_match_qdrant():
    with pytest.raises(
        ValueError,
        match="Embedding dimension does not match Qdrant vector size",
    ):
        Setting(
            **valid_settings(
                QDRANT_VECTOR_SIZE=768,
            )
        )

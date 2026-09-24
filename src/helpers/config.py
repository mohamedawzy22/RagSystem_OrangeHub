from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from services.vectordb.vector_db_enum import DistanceMetric

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"


class ChatModelSettings(BaseModel):
    provider: str
    model: str


class EmbeddingModelSettings(BaseModel):
    provider: str
    model: str
    dimension: int


class Setting(BaseSettings):
    APP_NAME: str
    APP_VERSION: str

    FILE_ALLOWED_TYPES: list[str]
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    MONGODB_URL: str
    MONGODB_DATABASE: str

    OPENROUTER_API_KEY: str | None = None
    OLLAMA_BASE_URL: str | None = None

    CHAT_MODELS: dict[str, ChatModelSettings]
    EMBEDDING_MODELS: dict[str, EmbeddingModelSettings]

    CHAT_PRIMARY_MODEL: str
    CHAT_FALLBACK_MODEL: str
    EMBEDDING_MODEL: str

    QDRANT_URL: str
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str
    QDRANT_VECTOR_SIZE: int
    QDRANT_DISTANCE: DistanceMetric

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        extra="ignore",
    )


def get_setting() -> Setting:
    return Setting()

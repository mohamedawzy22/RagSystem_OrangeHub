from pathlib import Path

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    LOG_LEVEL: str = "INFO"

    MODEL_WARMUP_ENABLED: bool = True
    MODEL_WARMUP_TIMEOUT_SECONDS: int = 30
    OLLAMA_KEEP_ALIVE: int = 300

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
    CHAT_FALLBACK_COOLDOWN_SECONDS: int = 60

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

    @model_validator(mode="after")
    def validate_models(self):
        if self.CHAT_PRIMARY_MODEL not in self.CHAT_MODELS:
            raise ValueError(
                f"Chat primary model '{self.CHAT_PRIMARY_MODEL}' is not configured"
            )

        if self.CHAT_FALLBACK_MODEL not in self.CHAT_MODELS:
            raise ValueError(
                f"Chat fallback model '{self.CHAT_FALLBACK_MODEL}' is not configured"
            )

        if self.CHAT_PRIMARY_MODEL == self.CHAT_FALLBACK_MODEL:
            raise ValueError("Chat primary and fallback models must be different")

        if self.EMBEDDING_MODEL not in self.EMBEDDING_MODELS:
            raise ValueError(
                f"Embedding model '{self.EMBEDDING_MODEL}' is not configured"
            )

        if self.CHAT_FALLBACK_COOLDOWN_SECONDS < 0:
            raise ValueError("CHAT_FALLBACK_COOLDOWN_SECONDS cannot be negative")

        if self.MODEL_WARMUP_TIMEOUT_SECONDS <= 0:
            raise ValueError("MODEL_WARMUP_TIMEOUT_SECONDS must be greater than zero")

        if self.OLLAMA_KEEP_ALIVE == 0 or self.OLLAMA_KEEP_ALIVE < -1:
            raise ValueError("OLLAMA_KEEP_ALIVE must be -1 or greater than zero")

        if self.QDRANT_VECTOR_SIZE <= 0:
            raise ValueError("QDRANT_VECTOR_SIZE must be greater than zero")

        embedding_dimension = self.EMBEDDING_MODELS[self.EMBEDDING_MODEL].dimension

        if embedding_dimension != self.QDRANT_VECTOR_SIZE:
            raise ValueError("Embedding dimension does not match Qdrant vector size")

        return self


def get_setting() -> Setting:
    return Setting()

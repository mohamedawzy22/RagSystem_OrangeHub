from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


def get_setting() -> Setting:
    return Setting()

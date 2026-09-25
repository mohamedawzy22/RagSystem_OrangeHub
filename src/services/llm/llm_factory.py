import logging

from helpers.config import (
    ChatModelSettings,
    EmbeddingModelSettings,
    Setting,
)

from .chat_interface import ChatModel
from .embedding_interface import EmbeddingModel
from .providers.ollama import (
    OllamaChatModel,
    OllamaEmbeddingModel,
)
from .providers.openrouter import (
    OpenRouterChatModel,
    OpenRouterEmbeddingModel,
)

logger = logging.getLogger("uvicorn")


class ModelFactory:
    def __init__(self, settings: Setting):
        self.settings = settings

    def create_chat_model(
        self,
        config: ChatModelSettings,
    ) -> ChatModel:

        logger.info(
            "Creating chat model: provider=%s | model=%s",
            config.provider,
            config.model,
        )

        if config.provider == "ollama":
            if not self.settings.OLLAMA_BASE_URL:
                raise ValueError("OLLAMA_BASE_URL is required")

            return OllamaChatModel(
                model_id=config.model,
                base_url=self.settings.OLLAMA_BASE_URL,
            )

        if config.provider == "openrouter":
            if not self.settings.OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY is required")

            return OpenRouterChatModel(
                model_id=config.model,
                api_key=self.settings.OPENROUTER_API_KEY,
            )

        raise ValueError(f"Unsupported chat provider: {config.provider}")

    def create_embedding_model(
        self,
        config: EmbeddingModelSettings,
    ) -> EmbeddingModel:

        logger.info(
            "Creating embedding model: provider=%s | model=%s | dimension=%s",
            config.provider,
            config.model,
            config.dimension,
        )

        if config.provider == "ollama":
            if not self.settings.OLLAMA_BASE_URL:
                raise ValueError("OLLAMA_BASE_URL is required")

            return OllamaEmbeddingModel(
                model_id=config.model,
                base_url=self.settings.OLLAMA_BASE_URL,
                dimension=config.dimension,
            )

        if config.provider == "openrouter":
            if not self.settings.OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY is required")

            return OpenRouterEmbeddingModel(
                model_id=config.model,
                api_key=self.settings.OPENROUTER_API_KEY,
                dimension=config.dimension,
            )

        raise ValueError(f"Unsupported embedding provider: {config.provider}")

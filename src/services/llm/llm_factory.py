from helpers.config import (
    ChatModelSettings,
    EmbeddingModelSettings,
    Setting,
)
from utils.logger import get_logger

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

logger = get_logger(__name__)


class ModelFactory:
    def __init__(
        self,
        settings: Setting,
    ):
        self.settings = settings

    def create_chat_model(
        self,
        config: ChatModelSettings,
    ) -> ChatModel:
        logger.debug(
            "Creating chat model: provider=%s | model=%s",
            config.provider,
            config.model,
        )

        if config.provider == "ollama":
            llm_config = self.settings.llm

            if not llm_config.ollama_base_url:
                raise ValueError(
                    "OLLAMA_BASE_URL is required",
                )

            return OllamaChatModel(
                model_id=config.model,
                base_url=llm_config.ollama_base_url,
                keep_alive=llm_config.ollama_keep_alive,
                timeout_seconds=(llm_config.ollama_request_timeout_seconds),
                max_concurrency=(llm_config.ollama_chat_max_concurrency),
                retry_config=self.settings.retry,
            )

        if config.provider == "openrouter":
            if not self.settings.OPENROUTER_API_KEY:
                raise ValueError(
                    "OPENROUTER_API_KEY is required",
                )

            return OpenRouterChatModel(
                model_id=config.model,
                api_key=self.settings.OPENROUTER_API_KEY,
            )

        raise ValueError(
            f"Unsupported chat provider: {config.provider}",
        )

    def create_embedding_model(
        self,
        config: EmbeddingModelSettings,
    ) -> EmbeddingModel:
        logger.debug(
            "Creating embedding model: provider=%s | model=%s | dimension=%s",
            config.provider,
            config.model,
            config.dimension,
        )

        if config.provider == "ollama":
            llm_config = self.settings.llm

            if not llm_config.ollama_base_url:
                raise ValueError(
                    "OLLAMA_BASE_URL is required",
                )

            return OllamaEmbeddingModel(
                model_id=config.model,
                base_url=llm_config.ollama_base_url,
                dimension=config.dimension,
                keep_alive=llm_config.ollama_keep_alive,
                timeout_seconds=(llm_config.ollama_request_timeout_seconds),
                max_concurrency=(llm_config.ollama_embedding_max_concurrency),
                retry_config=self.settings.retry,
            )

        if config.provider == "openrouter":
            if not self.settings.OPENROUTER_API_KEY:
                raise ValueError(
                    "OPENROUTER_API_KEY is required",
                )

            return OpenRouterEmbeddingModel(
                model_id=config.model,
                api_key=self.settings.OPENROUTER_API_KEY,
                dimension=config.dimension,
            )

        raise ValueError(
            f"Unsupported embedding provider: {config.provider}",
        )

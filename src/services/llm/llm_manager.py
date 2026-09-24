import logging

from .chat_interface import ChatModel
from .embedding_interface import EmbeddingModel
from .fallback_chat import FallbackChatModel
from .llm_factory import ModelFactory

logger = logging.getLogger("uvicorn")


class ModelManager:
    def __init__(
        self,
        factory: ModelFactory,
    ):
        self.factory = factory

        self._chat_models: dict[str, ChatModel] = {}
        self._embedding_models: dict[str, EmbeddingModel] = {}

        self._selected_chat_model: ChatModel | None = None
        self._selected_embedding_model: EmbeddingModel | None = None

    def load_models(self) -> None:

        logger.info("Loading chat models...")

        for name, config in self.factory.settings.CHAT_MODELS.items():
            self._chat_models[name] = self.factory.create_chat_model(config)

            logger.info(
                "Chat model loaded: %s",
                name,
            )

        logger.info("Loading embedding models...")

        for name, config in self.factory.settings.EMBEDDING_MODELS.items():
            self._embedding_models[name] = self.factory.create_embedding_model(config)

            logger.info(
                "Embedding model loaded: %s",
                name,
            )

        self._build_selected_chat_model()
        self._build_selected_embedding_model()

        logger.info(
            "Models loaded successfully: chat=%s | embedding=%s",
            len(self._chat_models),
            len(self._embedding_models),
        )

    def _build_selected_chat_model(self) -> None:

        primary_name = self.factory.settings.CHAT_PRIMARY_MODEL

        fallback_name = self.factory.settings.CHAT_FALLBACK_MODEL

        primary_model = self.get_chat_model(primary_name)

        fallback_model = self.get_chat_model(fallback_name)

        self._selected_chat_model = FallbackChatModel(
            primary=primary_model,
            fallback=fallback_model,
            primary_name=primary_name,
            fallback_name=fallback_name,
        )

        logger.info(
            "Selected chat models configured: primary=%s | fallback=%s",
            primary_name,
            fallback_name,
        )

    def _build_selected_embedding_model(self) -> None:

        embedding_name = self.factory.settings.EMBEDDING_MODEL

        self._selected_embedding_model = self.get_embedding_model(embedding_name)

        logger.info(
            "Selected embedding model configured: %s",
            embedding_name,
        )

    def get_chat_model(
        self,
        name: str,
    ) -> ChatModel:

        if name not in self._chat_models:
            logger.error(
                "Chat model not found: %s",
                name,
            )

            raise ValueError(f"Chat model '{name}' is not configured")

        return self._chat_models[name]

    def get_selected_chat_model(self) -> ChatModel:

        if self._selected_chat_model is None:
            raise RuntimeError("Selected chat model is not initialized")

        return self._selected_chat_model

    def get_embedding_model(
        self,
        name: str,
    ) -> EmbeddingModel:

        if name not in self._embedding_models:
            logger.error(
                "Embedding model not found: %s",
                name,
            )

            raise ValueError(f"Embedding model '{name}' is not configured")

        return self._embedding_models[name]

    def get_selected_embedding_model(
        self,
    ) -> EmbeddingModel:

        if self._selected_embedding_model is None:
            raise RuntimeError("Selected embedding model is not initialized")

        return self._selected_embedding_model

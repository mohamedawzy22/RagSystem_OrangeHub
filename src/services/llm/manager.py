import logging

from .chat_interface import ChatModel
from .embedding_interface import EmbeddingModel
from .factory import ModelFactory

logger = logging.getLogger("uvicorn")


class ModelManager:
    def __init__(self, factory: ModelFactory):
        self.factory = factory

        self._chat_models: dict[str, ChatModel] = {}
        self._embedding_models: dict[str, EmbeddingModel] = {}

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

        logger.info(
            "Models loaded successfully: chat=%s | embedding=%s",
            len(self._chat_models),
            len(self._embedding_models),
        )

    def get_chat_model(self, name: str) -> ChatModel:

        logger.info(
            "Getting chat model: %s",
            name,
        )

        if name not in self._chat_models:
            logger.error(
                "Chat model not found: %s",
                name,
            )

            raise ValueError(f"Chat model '{name}' is not configured")

        return self._chat_models[name]

    def get_embedding_model(
        self,
        name: str,
    ) -> EmbeddingModel:

        logger.info(
            "Getting embedding model: %s",
            name,
        )

        if name not in self._embedding_models:
            logger.error(
                "Embedding model not found: %s",
                name,
            )

            raise ValueError(f"Embedding model '{name}' is not configured")

        return self._embedding_models[name]

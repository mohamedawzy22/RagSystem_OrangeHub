from helpers.config import Setting
from services.vectordb.providers.qdrant import QdrantVectorDB
from services.vectordb.vector_db_enum import VectorDBProvider
from services.vectordb.vector_db_interface import VectorDB
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorDBFactory:
    def __init__(
        self,
        settings: Setting,
    ):
        self.settings = settings

    def create(
        self,
        provider: VectorDBProvider,
    ) -> VectorDB:
        if provider == VectorDBProvider.QDRANT:
            config = self.settings.qdrant

            return QdrantVectorDB(
                url=config.url,
                api_key=config.api_key,
                collection_name=config.collection_name,
                vector_size=config.vector_size,
                distance=config.distance,
                timeout_seconds=(config.request_timeout_seconds),
                retry_config=self.settings.retry,
            )

        raise ValueError(
            f"Unsupported vector database provider: {provider}",
        )

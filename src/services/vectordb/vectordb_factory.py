import logging

from helpers.config import Setting

from .providers.qdrant import QdrantVectorDB
from .vector_db_enum import VectorDBProvider
from .vector_db_interface import VectorDB

logger = logging.getLogger("uvicorn")


class VectorDBFactory:
    def __init__(self, settings: Setting):
        self.settings = settings

    def create(
        self,
        provider: VectorDBProvider,
    ) -> VectorDB:

        if provider == VectorDBProvider.QDRANT:
            if not self.settings.QDRANT_URL:
                raise ValueError("QDRANT_URL is required")

            return QdrantVectorDB(
                url=self.settings.QDRANT_URL,
                api_key=self.settings.QDRANT_API_KEY,
                collection_name=self.settings.QDRANT_COLLECTION_NAME,
                vector_size=self.settings.QDRANT_VECTOR_SIZE,
                distance=self.settings.QDRANT_DISTANCE,
            )

        raise ValueError(f"Unsupported vector database provider: {provider}")

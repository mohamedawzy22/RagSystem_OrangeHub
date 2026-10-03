from utils.logger import get_logger

from .vector_db_enum import VectorDBProvider
from .vector_db_interface import VectorDB
from .vectordb_factory import VectorDBFactory

logger = get_logger(__name__)


class VectorDBManager:
    def __init__(self, factory: VectorDBFactory):
        self.factory = factory
        self._databases: dict[str, VectorDB] = {}

    def load_databases(self) -> None:

        logger.info("Loading vector databases...")

        self._databases["default"] = self.factory.create(VectorDBProvider.QDRANT)

        logger.info("Vector databases loaded")

    def get_database(self, name: str) -> VectorDB:

        if name not in self._databases:
            raise ValueError(f"Vector database '{name}' is not configured")

        return self._databases[name]

    async def close(self) -> None:
        logger.info("Closing vector database clients...")

        for name, database in self._databases.items():
            try:
                await database.close()
                logger.debug(
                    "Closed vector database: %s",
                    name,
                )
            except Exception:
                logger.exception(
                    "Failed to close vector database: %s",
                    name,
                )

        logger.info("Vector database clients closed")

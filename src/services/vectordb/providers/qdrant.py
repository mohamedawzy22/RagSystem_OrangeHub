from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from utils.logger import get_logger

from ..vector_db_enum import DistanceMetric
from ..vector_db_interface import VectorDB

logger = get_logger(__name__)


class QdrantVectorDB(VectorDB):
    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int,
        distance: DistanceMetric,
        api_key: str | None = None,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        self.client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
        )

        logger.info(
            "Initialized Qdrant: collection=%s | vector_size=%s | distance=%s",
            collection_name,
            vector_size,
            distance.value,
        )

    async def close(self) -> None:
        logger.info(
            "Closing Qdrant client: collection=%s",
            self.collection_name,
        )

        await self.client.close()

    def _get_distance(self) -> models.Distance:
        distance_map = {
            DistanceMetric.COSINE: models.Distance.COSINE,
            DistanceMetric.EUCLID: models.Distance.EUCLID,
            DistanceMetric.DOT: models.Distance.DOT,
        }

        return distance_map[self.distance]

    async def create_collection(self) -> None:
        exists = await self.client.collection_exists(
            collection_name=self.collection_name
        )

        if exists:
            logger.info(
                "Qdrant collection already exists: %s",
                self.collection_name,
            )
            return

        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=self._get_distance(),
            ),
        )

        logger.info(
            "Qdrant collection created: %s | size=%s | distance=%s",
            self.collection_name,
            self.vector_size,
            self.distance.value,
        )

    async def delete_collection(self) -> None:
        await self.client.delete_collection(
            collection_name=self.collection_name,
        )

    async def upsert(
        self,
        vectors: list[dict],
    ) -> None:

        points = [
            models.PointStruct(
                id=item["id"],
                vector=item["vector"],
                payload=item.get("payload", {}),
            )
            for item in vectors
        ]

        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    async def search(
        self,
        vector: list[float],
        limit: int = 5,
    ) -> list[dict]:

        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=limit,
        )

        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
            }
            for result in results
        ]

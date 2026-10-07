import time
from uuid import NAMESPACE_URL, uuid5

from models import RAGMessage
from models.chunk_model import ChunkModel
from models.project_model import ProjectModel
from services.llm.embedding_interface import EmbeddingModel
from services.vectordb.vector_db_interface import VectorDB
from utils.logger import get_logger
from utils.metrics import (
    EMBEDDING_DURATION,
    INDEXING_BATCHES,
    INDEXING_CHUNKS,
    INDEXING_DURATION,
    INDEXING_FAILURES,
    VECTOR_STORE_DURATION,
)

logger = get_logger(__name__)


class IndexingService:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_db: VectorDB,
        project_model: ProjectModel,
        chunk_model: ChunkModel,
    ):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.project_model = project_model
        self.chunk_model = chunk_model

        logger.info(
            "Indexing service initialized successfully",
        )

    @staticmethod
    def _validate_batch_size(
        batch_size: int,
    ) -> None:
        if not isinstance(batch_size, int):
            raise TypeError(
                "Batch size must be an integer",
            )

        if batch_size <= 0:
            raise ValueError(
                "Batch size must be greater than zero",
            )

    @staticmethod
    def _build_qdrant_point_id(
        chunk_id,
    ) -> str:
        """
        Generate a deterministic UUID for a MongoDB chunk ID.

        The same chunk ID always generates the same UUID,
        allowing repeated indexing to update the same Qdrant point.
        """

        return str(
            uuid5(
                NAMESPACE_URL,
                f"chunk:{chunk_id}",
            )
        )

    async def index(
        self,
        project_id: str,
        batch_size: int = 32,
    ) -> dict:
        """
        Generate embeddings for project chunks in batches
        and store each batch in Qdrant.
        """

        self._validate_batch_size(
            batch_size,
        )

        logger.info(
            "Starting indexing: project_id=%s | batch_size=%s",
            project_id,
            batch_size,
        )

        indexing_start_time = time.perf_counter()

        try:
            project = await self.project_model.get_project_or_create_one(
                project_id=project_id,
            )

            logger.info(
                "Resolved project_id=%s to ObjectId=%s",
                project_id,
                project.id,
            )

            total_indexed_chunks = 0
            total_assets = set()
            total_batches = 0
            embedding_dimension = None

            async for chunks in self.chunk_model.iter_project_chunks(
                project_id=project.id,
                batch_size=batch_size,
            ):
                total_batches += 1

                logger.info(
                    "Processing batch %s: size=%s",
                    total_batches,
                    len(chunks),
                )

                texts = [chunk.chunk_text for chunk in chunks]

                logger.info(
                    "Generating embeddings: model=%s | batch=%s | chunks=%s",
                    getattr(
                        self.embedding_model,
                        "model_id",
                        "unknown",
                    ),
                    total_batches,
                    len(texts),
                )

                # --------------------------------------------
                # Embedding timing
                # --------------------------------------------
                embedding_start_time = time.perf_counter()

                try:
                    embeddings = await self.embedding_model.embed_documents(
                        texts,
                    )
                finally:
                    EMBEDDING_DURATION.observe(
                        time.perf_counter() - embedding_start_time,
                    )

                if len(embeddings) != len(chunks):
                    raise RuntimeError(
                        "Number of embeddings does not match number of chunks",
                    )

                if embeddings:
                    batch_dimension = len(
                        embeddings[0],
                    )

                    for embedding in embeddings:
                        if len(embedding) != batch_dimension:
                            raise RuntimeError(
                                "Embedding dimensions are inconsistent",
                            )

                    if embedding_dimension is None:
                        embedding_dimension = batch_dimension

                    elif batch_dimension != embedding_dimension:
                        raise RuntimeError(
                            "Embedding dimension changed between batches",
                        )

                    logger.info(
                        "Embeddings generated successfully: "
                        "batch=%s | count=%s | dimension=%s",
                        total_batches,
                        len(embeddings),
                        batch_dimension,
                    )

                vectors = [
                    {
                        "id": self._build_qdrant_point_id(
                            chunk.id,
                        ),
                        "vector": embedding,
                        "payload": {
                            "chunk_id": str(chunk.id),
                            "project_id": project_id,
                            "asset_id": str(
                                chunk.chunk_asset_id,
                            ),
                            "text": chunk.chunk_text,
                        },
                    }
                    for chunk, embedding in zip(
                        chunks,
                        embeddings,
                    )
                ]

                logger.info(
                    "Upserting vectors into Qdrant: "
                    "batch=%s | collection=%s | vectors=%s",
                    total_batches,
                    getattr(
                        self.vector_db,
                        "collection_name",
                        "unknown",
                    ),
                    len(vectors),
                )

                # --------------------------------------------
                # Vector store timing
                # --------------------------------------------
                vector_store_start_time = time.perf_counter()

                try:
                    await self.vector_db.upsert(
                        vectors,
                    )
                finally:
                    VECTOR_STORE_DURATION.observe(
                        time.perf_counter() - vector_store_start_time,
                    )

                # --------------------------------------------
                # Successful batch metrics
                # --------------------------------------------
                total_indexed_chunks += len(vectors)

                INDEXING_BATCHES.inc()
                INDEXING_CHUNKS.inc(
                    len(vectors),
                )

                total_assets.update(chunk.chunk_asset_id for chunk in chunks)

                logger.info(
                    "Batch indexing completed: batch=%s | indexed_chunks=%s",
                    total_batches,
                    total_indexed_chunks,
                )

            if total_indexed_chunks == 0:
                logger.info(
                    "No chunks found for project_id=%s",
                    project_id,
                )

                return {
                    "message": RAGMessage.NO_CHUNKS.value,
                    "project_id": project_id,
                    "assets": 0,
                    "chunks": 0,
                }

            logger.info(
                "Indexing completed successfully: "
                "project_id=%s | batches=%s | assets=%s | chunks=%s",
                project_id,
                total_batches,
                len(total_assets),
                total_indexed_chunks,
            )

            return {
                "message": RAGMessage.INDEX_SUCCESS.value,
                "project_id": project_id,
                "assets": len(total_assets),
                "chunks": total_indexed_chunks,
            }

        except Exception:
            INDEXING_FAILURES.inc()

            logger.exception(
                "Indexing failed for project_id=%s",
                project_id,
            )

            raise

        finally:
            # --------------------------------------------
            # Total indexing duration
            # --------------------------------------------
            INDEXING_DURATION.observe(
                time.perf_counter() - indexing_start_time,
            )

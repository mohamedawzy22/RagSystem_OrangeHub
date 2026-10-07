import time

from services.llm.embedding_interface import EmbeddingModel
from services.vectordb.vector_db_interface import VectorDB
from utils.logger import get_logger
from utils.metrics import (
    RETRIEVAL_DURATION,
    RETRIEVAL_EMBEDDING_DURATION,
    RETRIEVAL_FAILURES,
    RETRIEVAL_REQUESTS,
    RETRIEVAL_RESULTS,
    RETRIEVAL_VECTOR_SEARCH_DURATION,
)

logger = get_logger(__name__)


class RetrievalService:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_db: VectorDB,
    ):
        self.embedding_model = embedding_model
        self.vector_db = vector_db

        logger.info(
            "Retrieval service initialized successfully",
        )

    async def search(
        self,
        project_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Retrieve relevant chunks for a project.
        """

        logger.info(
            "Starting retrieval: project_id=%s | limit=%s",
            project_id,
            limit,
        )

        # Count every retrieval attempt,
        # including failed requests.
        RETRIEVAL_REQUESTS.inc()

        retrieval_start_time = time.perf_counter()

        try:
            # ------------------------------------------------
            # Query embedding timing
            # ------------------------------------------------
            embedding_start_time = time.perf_counter()

            try:
                query_embedding = await self.embedding_model.embed_text(
                    query,
                )
            finally:
                RETRIEVAL_EMBEDDING_DURATION.observe(
                    time.perf_counter() - embedding_start_time,
                )

            logger.info(
                "Query embedding generated: project_id=%s | dimension=%s",
                project_id,
                len(query_embedding),
            )

            # ------------------------------------------------
            # Vector search timing
            # ------------------------------------------------
            vector_search_start_time = time.perf_counter()

            try:
                results = await self.vector_db.search(
                    vector=query_embedding,
                    limit=limit,
                    project_id=project_id,
                )
            finally:
                RETRIEVAL_VECTOR_SEARCH_DURATION.observe(
                    time.perf_counter() - vector_search_start_time,
                )

            search_results = []

            for result in results:
                payload = result.get(
                    "payload",
                    {},
                )

                text = payload.get("text")

                if not text:
                    logger.warning(
                        "Search result has no text: project_id=%s",
                        project_id,
                    )
                    continue

                search_results.append(
                    {
                        "text": text,
                        "score": result["score"],
                    }
                )

            # Count only usable retrieved chunks.
            RETRIEVAL_RESULTS.inc(
                len(search_results),
            )

            logger.info(
                "Retrieval completed successfully: project_id=%s | results=%s",
                project_id,
                len(search_results),
            )

            return search_results

        except Exception:
            RETRIEVAL_FAILURES.inc()

            logger.exception(
                "Retrieval failed: project_id=%s",
                project_id,
            )

            raise

        finally:
            # ------------------------------------------------
            # Total retrieval timing
            # ------------------------------------------------
            RETRIEVAL_DURATION.observe(
                time.perf_counter() - retrieval_start_time,
            )

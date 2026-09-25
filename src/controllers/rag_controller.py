from uuid import NAMESPACE_URL, uuid5

from models import RAGMessage
from models.chunk_model import ChunkModel
from models.project_model import ProjectModel
from services.llm.chat_interface import ChatModel
from services.llm.embedding_interface import EmbeddingModel
from services.prompts.rag.english import (
    document_prompt,
    footer_prompt,
    system_prompt,
)
from services.vectordb.vectordb_manager import VectorDBManager
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGController:
    def __init__(
        self,
        chat_model: ChatModel,
        embedding_model: EmbeddingModel,
        vector_db_manager: VectorDBManager,
        db_client,
    ):
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.vector_db_manager = vector_db_manager
        self.db = db_client

        self.vector_db = vector_db_manager.get_database("default")

        logger.info("RAG controller initialized successfully")

    @staticmethod
    def _validate_project_id(
        project_id: str,
    ) -> None:
        if not isinstance(project_id, str):
            raise TypeError("Project ID must be a string")

        if not project_id.strip():
            raise ValueError("Project ID cannot be empty")

    @staticmethod
    def _validate_query(
        query: str,
    ) -> None:
        if not isinstance(query, str):
            raise TypeError("Query must be a string")

        if not query.strip():
            raise ValueError("Query cannot be empty")

    @staticmethod
    def _validate_limit(
        limit: int,
    ) -> None:
        if not isinstance(limit, int):
            raise TypeError("Limit must be an integer")

        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

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
    ) -> dict:
        """
        Generate embeddings for all chunks
        belonging to a project and store them in Qdrant.
        """

        self._validate_project_id(project_id)

        logger.info(
            "Starting indexing for project_id=%s",
            project_id,
        )

        try:
            # Resolve external project ID
            # to the MongoDB ObjectId.
            project_model = await ProjectModel.create_instance(db_client=self.db)

            project = await project_model.get_project_or_create_one(
                project_id=project_id
            )

            logger.info(
                "Resolved project_id=%s to ObjectId=%s",
                project_id,
                project.id,
            )

            # Get all chunks belonging to the project.
            chunk_model = await ChunkModel.create_instance(db_client=self.db)

            chunks = await chunk_model.get_project_chunks(
                project_id=project.id,
                page_no=1,
                page_size=100000,
            )

            logger.info(
                "Found %s chunks for project_id=%s",
                len(chunks),
                project_id,
            )

            if not chunks:
                return {
                    "message": RAGMessage.NO_CHUNKS.value,
                    "project_id": project_id,
                    "assets": 0,
                    "chunks": 0,
                }

            # Extract chunk texts.
            texts = [chunk.chunk_text for chunk in chunks]

            logger.info(
                "Generating embeddings: model=%s | chunks=%s",
                getattr(
                    self.embedding_model,
                    "model_id",
                    "unknown",
                ),
                len(texts),
            )

            # Generate embeddings.
            embeddings = await self.embedding_model.embed_documents(texts)

            if len(embeddings) != len(chunks):
                raise ValueError("Number of embeddings does not match number of chunks")

            # Validate embedding dimensions.
            if embeddings:
                embedding_dimension = len(embeddings[0])

                for embedding in embeddings:
                    if len(embedding) != embedding_dimension:
                        raise ValueError("Embedding dimensions are inconsistent")

                logger.info(
                    "Embeddings generated successfully: count=%s | dimension=%s",
                    len(embeddings),
                    embedding_dimension,
                )

            # Build Qdrant vectors.
            vectors = [
                {
                    "id": self._build_qdrant_point_id(chunk.id),
                    "vector": embedding,
                    "payload": {
                        "chunk_id": str(chunk.id),
                        "project_id": project_id,
                        "asset_id": str(chunk.chunk_asset_id),
                        "text": chunk.chunk_text,
                    },
                }
                for chunk, embedding in zip(
                    chunks,
                    embeddings,
                )
            ]

            logger.info(
                "Upserting vectors into Qdrant: collection=%s | vectors=%s",
                getattr(
                    self.vector_db,
                    "collection_name",
                    "unknown",
                ),
                len(vectors),
            )

            # Store embeddings in Qdrant.
            await self.vector_db.upsert(vectors)

            # Count unique assets represented
            # by the indexed chunks.
            asset_ids = {chunk.chunk_asset_id for chunk in chunks}

            logger.info(
                "Indexing completed successfully: "
                "project_id=%s | assets=%s | chunks=%s",
                project_id,
                len(asset_ids),
                len(vectors),
            )

            return {
                "message": RAGMessage.INDEX_SUCCESS.value,
                "project_id": project_id,
                "assets": len(asset_ids),
                "chunks": len(vectors),
            }

        except Exception:
            logger.exception(
                "Indexing failed for project_id=%s",
                project_id,
            )
            raise

    async def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Search for relevant chunks using vector similarity.
        """

        self._validate_query(query)
        self._validate_limit(limit)

        logger.info(
            "Starting search: limit=%s",
            limit,
        )

        try:
            # Generate embedding for the query.
            query_embedding = await self.embedding_model.embed_text(query)

            logger.info(
                "Query embedding generated: dimension=%s",
                len(query_embedding),
            )

            # Search in Qdrant.
            results = await self.vector_db.search(
                vector=query_embedding,
                limit=limit,
            )

            search_results = []

            for result in results:
                payload = result.get(
                    "payload",
                    {},
                )

                text = payload.get("text")

                if not text:
                    logger.warning("Search result has no text")
                    continue

                search_results.append(
                    {
                        "text": text,
                        "score": result["score"],
                    }
                )

            logger.info(
                "Search completed successfully: results=%s",
                len(search_results),
            )

            return search_results

        except Exception:
            logger.exception("Search failed")
            raise

    @staticmethod
    def _build_prompt(
        query: str,
        results: list[dict],
    ) -> tuple[str, list[dict]]:
        system_text = system_prompt.substitute()

        documents = [
            {
                "text": result["text"],
                "score": result["score"],
            }
            for result in results
        ]

        context = "\n\n".join(
            document_prompt.substitute(
                doc_num=index,
                chunk_text=document["text"],
            )
            for index, document in enumerate(
                documents,
                start=1,
            )
        )

        footer_text = footer_prompt.substitute(
            query=query,
        )

        full_prompt = "\n\n".join(
            part
            for part in (
                system_text,
                context,
                footer_text,
            )
            if part
        )

        return full_prompt, documents

    async def generate(
        self,
        query: str,
        limit: int = 5,
    ) -> dict:

        self._validate_query(query)
        self._validate_limit(limit)

        logger.info(
            "Starting generation: limit=%s",
            limit,
        )

        try:
            results = await self.search(
                query=query,
                limit=limit,
            )

            full_prompt, documents = self._build_prompt(
                query=query,
                results=results,
            )

            if not results:
                logger.info(
                    "No relevant documents found",
                )

                return {
                    "query": query,
                    "answer": (RAGMessage.NO_RELEVANT_INFORMATION.value),
                    "full_prompt": full_prompt,
                    "documents": [],
                }

            response = await self.chat_model.generate(
                prompt=full_prompt,
            )

            logger.info(
                "Generation completed successfully",
            )

            return {
                "query": query,
                "answer": response,
                "full_prompt": full_prompt,
                "documents": documents,
            }

        except Exception:
            logger.exception(
                "Generation failed",
            )
            raise

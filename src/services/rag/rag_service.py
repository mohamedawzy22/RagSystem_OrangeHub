from models import RAGMessage
from services.llm.chat_interface import ChatModel
from services.prompts.rag.english import (
    document_prompt,
    footer_prompt,
    system_prompt,
)
from services.rag.indexing_service import IndexingService
from services.rag.retrieval_service import RetrievalService
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGService:
    def __init__(
        self,
        chat_model: ChatModel,
        retrieval_service: RetrievalService,
        indexing_service: IndexingService,
    ):
        self.chat_model = chat_model
        self.retrieval_service = retrieval_service
        self.indexing_service = indexing_service

        logger.info("RAG service initialized successfully")

    async def index(
        self,
        project_id: str,
        batch_size: int = 32,
    ) -> dict:
        return await self.indexing_service.index(
            project_id=project_id,
            batch_size=batch_size,
        )

    async def search(
        self,
        project_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        return await self.retrieval_service.search(
            project_id=project_id,
            query=query,
            limit=limit,
        )

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
        project_id: str,
        query: str,
        limit: int = 5,
    ) -> dict:
        """
        Search for relevant chunks and generate
        an answer using the retrieved context.
        """

        logger.info(
            "Starting generation: project_id=%s | limit=%s",
            project_id,
            limit,
        )

        try:
            results = await self.retrieval_service.search(
                project_id=project_id,
                query=query,
                limit=limit,
            )

            full_prompt, documents = self._build_prompt(
                query=query,
                results=results,
            )

            if not results:
                logger.info(
                    "No relevant documents found: project_id=%s",
                    project_id,
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
                "Generation completed successfully: project_id=%s",
                project_id,
            )

            return {
                "query": query,
                "answer": response,
                "full_prompt": full_prompt,
                "documents": documents,
            }

        except Exception:
            logger.exception(
                "Generation failed: project_id=%s",
                project_id,
            )
            raise

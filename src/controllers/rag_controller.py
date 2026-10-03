from services.rag.rag_service import RAGService


class RAGController:
    def __init__(
        self,
        rag_service: RAGService,
    ):
        self.rag_service = rag_service

    async def index(
        self,
        project_id: str,
        batch_size: int = 32,
    ) -> dict:
        return await self.rag_service.index(
            project_id=project_id,
            batch_size=batch_size,
        )

    async def search(
        self,
        project_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        return await self.rag_service.search(
            project_id=project_id,
            query=query,
            limit=limit,
        )

    async def generate(
        self,
        project_id: str,
        query: str,
        limit: int = 5,
    ) -> dict:
        return await self.rag_service.generate(
            project_id=project_id,
            query=query,
            limit=limit,
        )

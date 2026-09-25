from fastapi import APIRouter, HTTPException, Request

from routes.schemes.rag import (
    GenerateRequest,
    GenerateResponse,
    SearchRequest,
    SearchResponse,
)
from utils.logger import get_logger

logger = get_logger(__name__)


rag_router = APIRouter(
    prefix="/api/v1/rag",
    tags=["api_v1", "rag"],
)


@rag_router.post("/index/{project_id}")
async def index(
    request: Request,
    project_id: str,
):
    logger.info(
        "RAG INDEX ROUTE REACHED: project_id=%s",
        project_id,
    )

    try:
        return await request.app.rag_controller.index(
            project_id=project_id,
        )

    except ValueError as exc:
        logger.error(
            "RAG indexing validation error: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception(
            "RAG indexing failed: project_id=%s",
            project_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to index project",
        )


@rag_router.post(
    "/search",
    response_model=SearchResponse,
)
async def search(
    request: Request,
    search_request: SearchRequest,
):
    try:
        results = await request.app.rag_controller.search(
            query=search_request.query,
            limit=search_request.limit,
        )

        return SearchResponse(
            query=search_request.query,
            results=results,
        )

    except ValueError as exc:
        logger.error(
            "RAG search validation error: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception(
            "RAG search failed",
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to search documents",
        )


@rag_router.post(
    "/generate",
    response_model=GenerateResponse,
)
async def generate(
    request: Request,
    generate_request: GenerateRequest,
):
    try:
        result = await request.app.rag_controller.generate(
            query=generate_request.query,
            limit=generate_request.limit,
        )

        return GenerateResponse(
            **result,
        )

    except ValueError as exc:
        logger.error(
            "RAG generation validation error: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception(
            "RAG generation failed",
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate response",
        )

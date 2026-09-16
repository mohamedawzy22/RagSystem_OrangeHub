import logging

from fastapi import APIRouter, HTTPException, Request, status

from services.llm.llm_enum import ChatModelName

from .schemes.chat import ChatRequest, ChatResponse

logger = logging.getLogger("uvicorn.error")


chat_router = APIRouter(
    prefix="/api/v1/chat",
    tags=["api_v1", "chat"],
)


@chat_router.post(
    "/generate",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_text(
    request: Request,
    chat_request: ChatRequest,
):
    model_name: ChatModelName = chat_request.model

    logger.info(
        "Chat generation requested: model=%s",
        model_name.value,
    )

    try:
        model = request.app.model_manager.get_chat_model(model_name.value)

        response = await model.generate(
            prompt=chat_request.prompt,
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens,
        )

        logger.info(
            "Chat generation completed: model=%s",
            model_name.value,
        )

        return ChatResponse(
            model=model_name,
            response=response,
        )

    except ValueError as exc:
        logger.warning(
            "Chat model is not configured: model=%s",
            model_name.value,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Chat generation failed: model=%s",
            model_name.value,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response",
        ) from exc

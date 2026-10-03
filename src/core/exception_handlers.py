from fastapi import Request
from fastapi.responses import JSONResponse

from core.exceptions import ApplicationError
from utils.logger import get_logger

logger = get_logger(__name__)


async def application_error_handler(
    request: Request,
    exc: ApplicationError,
) -> JSONResponse:
    logger.warning(
        "Application error: method=%s path=%s status=%s detail=%s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.detail,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
        },
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception(
        "Unhandled application exception: method=%s path=%s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
        },
    )

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from helpers.config import RetryConfig
from utils.logger import get_logger

T = TypeVar("T")

logger = get_logger(__name__)


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    should_retry: Callable[[Exception], bool],
    config: RetryConfig,
    operation_name: str,
) -> T:
    last_exception: Exception | None = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            return await operation()

        except Exception as exc:
            last_exception = exc

            if attempt >= config.max_attempts or not should_retry(exc):
                raise

            exponential_delay = min(
                config.initial_delay_seconds * (2 ** (attempt - 1)),
                config.max_delay_seconds,
            )

            jitter = random.uniform(
                0,
                exponential_delay * 0.25,
            )

            delay = exponential_delay + jitter

            logger.warning(
                "Transient provider error: "
                "operation=%s | attempt=%s/%s | "
                "retry_in=%.2fs | error=%s",
                operation_name,
                attempt,
                config.max_attempts,
                delay,
                type(exc).__name__,
            )

            await asyncio.sleep(delay)

    if last_exception is not None:
        raise last_exception

    raise RuntimeError(
        f"Retry operation failed: {operation_name}",
    )

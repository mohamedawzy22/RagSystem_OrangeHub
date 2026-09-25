import time
from collections.abc import AsyncIterator

from utils.logger import get_logger

from .chat_interface import ChatModel

logger = get_logger(__name__)


class FallbackChatModel(ChatModel):
    def __init__(
        self,
        primary: ChatModel,
        fallback: ChatModel,
        primary_name: str,
        fallback_name: str,
        cooldown_seconds: int = 60,
    ):
        self.primary = primary
        self.fallback = fallback

        self.primary_name = primary_name
        self.fallback_name = fallback_name

        self.cooldown_seconds = cooldown_seconds
        self._primary_failed_at: float | None = None

        logger.info(
            "Chat fallback configured: primary=%s | fallback=%s | cooldown=%ss",
            self.primary_name,
            self.fallback_name,
            self.cooldown_seconds,
        )

    def set_primary_unavailable(self) -> None:
        self._mark_primary_failed()

        logger.warning(
            "Primary chat model marked unavailable: %s",
            self.primary_name,
        )

    def _primary_available(self) -> bool:
        if self._primary_failed_at is None:
            return True

        elapsed = time.monotonic() - self._primary_failed_at

        if elapsed >= self.cooldown_seconds:
            logger.info(
                "Primary chat model cooldown expired: %s",
                self.primary_name,
            )
            return True

        logger.debug(
            "Skipping primary chat model during cooldown: %s",
            self.primary_name,
        )
        return False

    def _mark_primary_failed(self) -> None:
        self._primary_failed_at = time.monotonic()

    def _mark_primary_success(self) -> None:
        self._primary_failed_at = None

    async def health_check(self) -> bool:
        try:
            primary_healthy = await self.primary.health_check()
        except Exception:
            logger.exception(
                "Primary chat model health check failed: %s",
                self.primary_name,
            )
            primary_healthy = False

        if primary_healthy:
            logger.info(
                "Primary chat model is healthy: %s",
                self.primary_name,
            )
            return True

        logger.warning(
            "Primary chat model is unavailable: %s",
            self.primary_name,
        )

        try:
            fallback_healthy = await self.fallback.health_check()
        except Exception:
            logger.exception(
                "Fallback chat model health check failed: %s",
                self.fallback_name,
            )
            return False

        if fallback_healthy:
            logger.info(
                "Fallback chat model is healthy: %s",
                self.fallback_name,
            )

        return fallback_healthy

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:

        if self._primary_available():
            try:
                logger.info(
                    "Using primary chat model: %s",
                    self.primary_name,
                )

                response = await self.primary.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                self._mark_primary_success()

                logger.debug(
                    "Primary chat model succeeded: %s",
                    self.primary_name,
                )

                return response

            except Exception:  # noqa: BLE001
                self._mark_primary_failed()

                logger.warning(
                    "Primary chat model failed: %s | fallback=%s",
                    self.primary_name,
                    self.fallback_name,
                )

        else:
            logger.warning(
                "Primary chat model is in cooldown: %s | using fallback=%s",
                self.primary_name,
                self.fallback_name,
            )

        try:
            logger.info(
                "Using fallback chat model: %s",
                self.fallback_name,
            )

            return await self.fallback.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        except Exception:
            logger.exception(
                "Fallback chat model failed: %s",
                self.fallback_name,
            )
            raise

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:

        yielded_content = False

        if self._primary_available():
            try:
                logger.info(
                    "Starting stream with primary chat model: %s",
                    self.primary_name,
                )

                async for chunk in self.primary.stream(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    yielded_content = True
                    yield chunk

                self._mark_primary_success()
                return

            except Exception:
                self._mark_primary_failed()

                if yielded_content:
                    logger.exception(
                        "Primary stream failed after partial response: %s",
                        self.primary_name,
                    )
                    raise

                logger.warning(
                    "Primary stream failed before response: %s | fallback=%s",
                    self.primary_name,
                    self.fallback_name,
                )

        else:
            logger.warning(
                "Primary chat model is in cooldown: %s | using fallback=%s",
                self.primary_name,
                self.fallback_name,
            )

        try:
            logger.info(
                "Starting stream with fallback chat model: %s",
                self.fallback_name,
            )

            async for chunk in self.fallback.stream(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield chunk

        except Exception:
            logger.exception(
                "Fallback chat stream failed: %s",
                self.fallback_name,
            )
            raise

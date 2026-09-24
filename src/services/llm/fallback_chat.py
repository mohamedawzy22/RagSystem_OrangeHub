import logging
from collections.abc import AsyncIterator

from .chat_interface import ChatModel

logger = logging.getLogger("uvicorn")


class FallbackChatModel(ChatModel):
    def __init__(
        self,
        primary: ChatModel,
        fallback: ChatModel,
        primary_name: str,
        fallback_name: str,
    ):
        self.primary = primary
        self.fallback = fallback

        self.primary_name = primary_name
        self.fallback_name = fallback_name

        logger.info(
            "Chat fallback configured: primary=%s | fallback=%s",
            self.primary_name,
            self.fallback_name,
        )

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:

        try:
            logger.info(
                "Using primary chat model: %s",
                self.primary_name,
            )

            return await self.primary.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        except Exception:  # noqa: BLE001
            logger.warning(
                "Primary chat model failed: %s. Trying fallback model: %s",
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
            logger.exception("Both primary and fallback chat models failed")
            raise

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:

        yielded_content = False

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

            return

        except Exception:
            if yielded_content:
                logger.exception(
                    "Primary stream failed after partial response: %s",
                    self.primary_name,
                )
                raise

            logger.warning(
                "Primary stream failed before response. Trying fallback model: %s",
                self.primary_name,
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
            logger.exception("Both primary and fallback chat streams failed")
            raise

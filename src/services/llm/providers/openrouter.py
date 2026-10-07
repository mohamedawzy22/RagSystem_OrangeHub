import asyncio
import random
from collections.abc import AsyncIterator

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

from core.retry import retry_async
from helpers.config import RetryConfig
from utils.logger import get_logger
from utils.metrics import (
    LLM_INPUT_TOKENS,
    LLM_OUTPUT_TOKENS,
    LLM_TOTAL_TOKENS,
)

from ..chat_interface import ChatModel
from ..embedding_interface import EmbeddingModel

logger = get_logger(__name__)


_RETRYABLE_STATUS_CODES = {
    408,
    409,
    429,
}


def _is_retryable_openrouter_error(
    exc: Exception,
) -> bool:
    """
    Return whether an OpenRouter/OpenAI SDK error is retryable.
    """

    if isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        ),
    ):
        return True

    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS_CODES or exc.status_code >= 500

    return False


def _calculate_retry_delay(
    retry_config: RetryConfig,
    attempt: int,
) -> float:
    """
    Calculate exponential backoff with jitter.
    """

    exponential_delay = min(
        retry_config.initial_delay_seconds * (2 ** (attempt - 1)),
        retry_config.max_delay_seconds,
    )

    jitter = exponential_delay * 0.25 * random.random()

    return exponential_delay + jitter


class OpenRouterChatModel(ChatModel):
    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: str,
        timeout_seconds: int = 120,
        max_tokens: int = 300,
        max_concurrency: int = 1,
        retry_config: RetryConfig | None = None,
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens

        self.timeout_seconds = timeout_seconds
        self.max_concurrency = max_concurrency
        self.retry_config = retry_config

        self._semaphore = asyncio.Semaphore(
            max_concurrency,
        )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )

        logger.info(
            "Initialized OpenRouter chat model: "
            "model=%s | timeout=%ss | "
            "max_tokens=%s | max_concurrency=%s",
            self.model_id,
            self.timeout_seconds,
            self.max_tokens,
            self.max_concurrency,
        )

    def _record_token_usage(
        self,
        response,
    ) -> bool:
        """
        Record token usage returned by OpenRouter.
        """

        usage = getattr(
            response,
            "usage",
            None,
        )

        if usage is None:
            return False

        input_tokens = getattr(
            usage,
            "prompt_tokens",
            0,
        )

        output_tokens = getattr(
            usage,
            "completion_tokens",
            0,
        )

        total_tokens = getattr(
            usage,
            "total_tokens",
            input_tokens + output_tokens,
        )

        LLM_INPUT_TOKENS.labels(
            provider="openrouter",
            model=self.model_id,
        ).inc(input_tokens)

        LLM_OUTPUT_TOKENS.labels(
            provider="openrouter",
            model=self.model_id,
        ).inc(output_tokens)

        LLM_TOTAL_TOKENS.labels(
            provider="openrouter",
            model=self.model_id,
        ).inc(total_tokens)

        logger.debug(
            "OpenRouter token usage: model=%s | input=%s | output=%s | total=%s",
            self.model_id,
            input_tokens,
            output_tokens,
            total_tokens,
        )

        return True

    async def _generate_once(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
            max_tokens=effective_max_tokens,
            extra_body={
                "usage": {
                    "include": True,
                },
            },
        )

        self._record_token_usage(
            response,
        )

        content = response.choices[0].message.content

        return content or ""

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        logger.info(
            "Generating text with OpenRouter model=%s",
            self.model_id,
        )

        async with self._semaphore:
            try:
                if self.retry_config is None:
                    return await self._generate_once(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                return await retry_async(
                    lambda: self._generate_once(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    should_retry=_is_retryable_openrouter_error,
                    config=self.retry_config,
                    operation_name=(f"openrouter.chat:{self.model_id}"),
                )

            except APITimeoutError:
                logger.error(
                    "OpenRouter generation timed out: model=%s",
                    self.model_id,
                )
                raise

            except Exception:
                logger.exception(
                    "OpenRouter generation failed: model=%s",
                    self.model_id,
                )
                raise

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        logger.info(
            "Starting OpenRouter stream: model=%s",
            self.model_id,
        )

        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        async with self._semaphore:
            yielded_content = False
            usage_recorded = False

            attempts = (
                self.retry_config.max_attempts if self.retry_config is not None else 1
            )

            for attempt in range(
                1,
                attempts + 1,
            ):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_id,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        temperature=temperature,
                        max_tokens=effective_max_tokens,
                        stream=True,
                        extra_body={
                            "usage": {
                                "include": True,
                            },
                        },
                    )

                    async for chunk in response:
                        if chunk.usage is not None:
                            usage_recorded = self._record_token_usage(
                                chunk,
                            )
                            continue

                        if not chunk.choices:
                            continue

                        content = chunk.choices[0].delta.content

                        if content:
                            yielded_content = True
                            yield content

                    logger.info(
                        "OpenRouter stream completed: model=%s",
                        self.model_id,
                    )

                    if not usage_recorded:
                        logger.debug(
                            "OpenRouter stream completed without token usage: model=%s",
                            self.model_id,
                        )

                    return

                except Exception as exc:
                    should_stop_retrying = (
                        yielded_content
                        or self.retry_config is None
                        or attempt >= attempts
                        or not _is_retryable_openrouter_error(exc)
                    )

                    if should_stop_retrying:
                        logger.exception(
                            "OpenRouter stream failed: model=%s",
                            self.model_id,
                        )
                        raise

                    delay = _calculate_retry_delay(
                        retry_config=self.retry_config,
                        attempt=attempt,
                    )

                    logger.warning(
                        "Retrying OpenRouter stream: "
                        "model=%s | attempt=%s/%s | "
                        "retry_in=%.2fs",
                        self.model_id,
                        attempt,
                        attempts,
                        delay,
                    )

                    await asyncio.sleep(delay)

    async def health_check(self) -> bool:
        try:
            await self.client.models.retrieve(
                self.model_id,
            )

            logger.info(
                "OpenRouter chat model is healthy: %s",
                self.model_id,
            )

            return True

        except Exception:
            logger.exception(
                "OpenRouter chat model health check failed: %s",
                self.model_id,
            )
            return False

    async def close(self) -> None:
        await self.client.close()


class OpenRouterEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_id: str,
        api_key: str,
        dimension: int,
        base_url: str,
        timeout_seconds: int = 120,
        max_concurrency: int = 2,
        retry_config: RetryConfig | None = None,
    ):
        self.model_id = model_id
        self.dimension = dimension

        self.timeout_seconds = timeout_seconds
        self.max_concurrency = max_concurrency
        self.retry_config = retry_config

        self._semaphore = asyncio.Semaphore(
            max_concurrency,
        )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )

        logger.info(
            "Initialized OpenRouter embedding model: "
            "model=%s | dimension=%s | "
            "timeout=%ss | max_concurrency=%s",
            self.model_id,
            self.dimension,
            self.timeout_seconds,
            self.max_concurrency,
        )

    async def _embed_once(
        self,
        text: str,
    ) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.model_id,
            input=text,
        )

        embedding = response.data[0].embedding

        self._validate_dimension(
            embedding,
        )

        return embedding

    async def _embed_documents_once(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.model_id,
            input=texts,
        )

        embeddings = [item.embedding for item in response.data]

        for embedding in embeddings:
            self._validate_dimension(
                embedding,
            )

        return embeddings

    def _validate_dimension(
        self,
        embedding: list[float],
    ) -> None:
        if len(embedding) != self.dimension:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"expected={self.dimension} | "
                f"actual={len(embedding)}",
            )

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:
        logger.info(
            "Creating embedding with OpenRouter model=%s",
            self.model_id,
        )

        async with self._semaphore:
            try:
                if self.retry_config is None:
                    return await self._embed_once(
                        text,
                    )

                return await retry_async(
                    lambda: self._embed_once(
                        text,
                    ),
                    should_retry=_is_retryable_openrouter_error,
                    config=self.retry_config,
                    operation_name=(f"openrouter.embed:{self.model_id}"),
                )

            except APITimeoutError:
                logger.error(
                    "OpenRouter embedding timed out: model=%s",
                    self.model_id,
                )
                raise

            except Exception:
                logger.exception(
                    "OpenRouter embedding failed: model=%s",
                    self.model_id,
                )
                raise

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        logger.info(
            "Creating document embeddings: model=%s | documents=%s",
            self.model_id,
            len(texts),
        )

        if not texts:
            return []

        async with self._semaphore:
            try:
                if self.retry_config is None:
                    return await self._embed_documents_once(
                        texts,
                    )

                return await retry_async(
                    lambda: self._embed_documents_once(
                        texts,
                    ),
                    should_retry=_is_retryable_openrouter_error,
                    config=self.retry_config,
                    operation_name=(f"openrouter.embed_documents:{self.model_id}"),
                )

            except APITimeoutError:
                logger.error(
                    "OpenRouter document embeddings timed out: model=%s",
                    self.model_id,
                )
                raise

            except Exception:
                logger.exception(
                    "OpenRouter document embeddings failed: model=%s",
                    self.model_id,
                )
                raise

    async def health_check(self) -> bool:
        try:
            await self.client.models.retrieve(
                self.model_id,
            )

            logger.info(
                "OpenRouter embedding model is healthy: %s",
                self.model_id,
            )

            return True

        except Exception:
            logger.exception(
                "OpenRouter embedding model health check failed: %s",
                self.model_id,
            )
            return False

    async def close(self) -> None:
        await self.client.close()

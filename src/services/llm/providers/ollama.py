import asyncio
import random

from ollama import AsyncClient, ResponseError

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
    429,
}


def _is_retryable_ollama_error(
    exc: Exception,
) -> bool:
    if isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        ),
    ):
        return True

    if isinstance(exc, ResponseError):
        return exc.status_code in _RETRYABLE_STATUS_CODES or exc.status_code >= 500

    return False


class OllamaChatModel(ChatModel):
    def __init__(
        self,
        model_id: str,
        base_url: str,
        keep_alive: int = 300,
        timeout_seconds: int = 120,
        max_tokens: int = 300,
        max_concurrency: int = 1,
        retry_config: RetryConfig | None = None,
    ):
        self.model_id = model_id
        self.keep_alive = keep_alive
        self.max_tokens = max_tokens

        self.timeout_seconds = timeout_seconds
        self.max_concurrency = max_concurrency

        self.retry_config = retry_config

        self._semaphore = asyncio.Semaphore(
            max_concurrency,
        )

        self.client = AsyncClient(
            host=base_url,
            timeout=timeout_seconds,
        )

        logger.info(
            "Initialized Ollama chat model: "
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
        Record token usage returned by Ollama.

        Returns True when token usage was available.
        """

        input_tokens = response.get(
            "prompt_eval_count",
        )
        output_tokens = response.get(
            "eval_count",
        )

        if input_tokens is None and output_tokens is None:
            return False

        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0
        total_tokens = input_tokens + output_tokens

        LLM_INPUT_TOKENS.labels(
            provider="ollama",
            model=self.model_id,
        ).inc(input_tokens)

        LLM_OUTPUT_TOKENS.labels(
            provider="ollama",
            model=self.model_id,
        ).inc(output_tokens)

        LLM_TOTAL_TOKENS.labels(
            provider="ollama",
            model=self.model_id,
        ).inc(total_tokens)

        logger.debug(
            "Ollama token usage: model=%s | input=%s | output=%s | total=%s",
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
        options = {
            "temperature": temperature,
        }

        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        options["num_predict"] = effective_max_tokens

        async with asyncio.timeout(
            self.timeout_seconds,
        ):
            response = await self.client.chat(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                options=options,
                keep_alive=self.keep_alive,
            )

        self._record_token_usage(
            response,
        )

        return response["message"]["content"]

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        logger.info(
            "Generating text with Ollama model=%s",
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
                    should_retry=_is_retryable_ollama_error,
                    config=self.retry_config,
                    operation_name=(f"ollama.chat:{self.model_id}"),
                )

            except TimeoutError:
                logger.error(
                    "Ollama generation timed out: model=%s",
                    self.model_id,
                )
                raise

            except Exception:
                logger.exception(
                    "Ollama generation failed: model=%s",
                    self.model_id,
                )
                raise

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ):
        logger.info(
            "Starting Ollama stream: model=%s",
            self.model_id,
        )

        options = {
            "temperature": temperature,
        }

        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        options["num_predict"] = effective_max_tokens

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
                    async with asyncio.timeout(
                        self.timeout_seconds,
                    ):
                        response = await self.client.chat(
                            model=self.model_id,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            options=options,
                            stream=True,
                            keep_alive=self.keep_alive,
                        )

                        async for chunk in response:
                            content = chunk["message"]["content"]

                            if content:
                                yielded_content = True
                                yield content

                            if not usage_recorded:
                                usage_recorded = self._record_token_usage(
                                    chunk,
                                )

                    logger.info(
                        "Ollama stream completed: model=%s",
                        self.model_id,
                    )
                    return

                except Exception as exc:
                    if (
                        yielded_content
                        or self.retry_config is None
                        or attempt >= attempts
                        or not _is_retryable_ollama_error(exc)
                    ):
                        logger.exception(
                            "Ollama stream failed: model=%s",
                            self.model_id,
                        )
                        raise

                    exponential_delay = min(
                        self.retry_config.initial_delay_seconds * (2 ** (attempt - 1)),
                        self.retry_config.max_delay_seconds,
                    )

                    jitter = exponential_delay * 0.25 * random.random()

                    delay = exponential_delay + jitter

                    logger.warning(
                        "Retrying Ollama stream: "
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
            await self.client.show(
                self.model_id,
            )

            logger.info(
                "Ollama chat model is healthy: %s",
                self.model_id,
            )

            return True

        except Exception:
            logger.exception(
                "Ollama chat model health check failed: %s",
                self.model_id,
            )
            return False

    async def close(self) -> None:
        await self.client.close()


class OllamaEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_id: str,
        base_url: str,
        dimension: int,
        keep_alive: int = 300,
        timeout_seconds: int = 120,
        max_concurrency: int = 2,
        retry_config: RetryConfig | None = None,
    ):
        self.model_id = model_id
        self.dimension = dimension
        self.keep_alive = keep_alive

        self.timeout_seconds = timeout_seconds
        self.max_concurrency = max_concurrency

        self.retry_config = retry_config

        self._semaphore = asyncio.Semaphore(
            max_concurrency,
        )

        self.client = AsyncClient(
            host=base_url,
            timeout=timeout_seconds,
        )

        logger.info(
            "Initialized Ollama embedding model: "
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
        async with asyncio.timeout(
            self.timeout_seconds,
        ):
            response = await self.client.embed(
                model=self.model_id,
                input=text,
                keep_alive=self.keep_alive,
            )

        return response["embeddings"][0]

    async def _embed_documents_once(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        async with asyncio.timeout(
            self.timeout_seconds,
        ):
            response = await self.client.embed(
                model=self.model_id,
                input=texts,
                keep_alive=self.keep_alive,
            )

        return response["embeddings"]

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:
        logger.info(
            "Creating embedding with Ollama model=%s",
            self.model_id,
        )

        async with self._semaphore:
            try:
                if self.retry_config is None:
                    return await self._embed_once(text)

                return await retry_async(
                    lambda: self._embed_once(text),
                    should_retry=_is_retryable_ollama_error,
                    config=self.retry_config,
                    operation_name=(f"ollama.embed:{self.model_id}"),
                )

            except Exception:
                logger.exception(
                    "Ollama embedding failed: model=%s",
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
                    should_retry=_is_retryable_ollama_error,
                    config=self.retry_config,
                    operation_name=(f"ollama.embed_documents:{self.model_id}"),
                )

            except Exception:
                logger.exception(
                    "Ollama document embeddings failed: model=%s",
                    self.model_id,
                )
                raise

    async def health_check(self) -> bool:
        try:
            await self.client.show(
                self.model_id,
            )

            logger.info(
                "Ollama embedding model is healthy: %s",
                self.model_id,
            )

            return True

        except Exception:
            logger.exception(
                "Ollama embedding model health check failed: %s",
                self.model_id,
            )
            return False

    async def close(self) -> None:
        await self.client.close()

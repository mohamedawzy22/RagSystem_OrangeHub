import logging

from openai import AsyncOpenAI

from ..chat_interface import ChatModel
from ..embedding_interface import EmbeddingModel

logger = logging.getLogger("uvicorn")


class OpenRouterChatModel(ChatModel):
    def __init__(
        self,
        model_id: str,
        api_key: str,
    ):
        self.model_id = model_id

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        logger.info(
            "Initialized OpenRouter chat model: %s",
            self.model_id,
        )

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
                max_tokens=max_tokens,
            )

            logger.info(
                "OpenRouter generation completed: model=%s",
                self.model_id,
            )

            return response.choices[0].message.content

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
    ):

        logger.info(
            "Starting OpenRouter stream: model=%s",
            self.model_id,
        )

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
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in response:
                content = chunk.choices[0].delta.content

                if content:
                    yield content

            logger.info(
                "OpenRouter stream completed: model=%s",
                self.model_id,
            )

        except Exception:
            logger.exception(
                "OpenRouter stream failed: model=%s",
                self.model_id,
            )
            raise


class OpenRouterEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_id: str,
        api_key: str,
        dimension: int,
    ):
        self.model_id = model_id
        self.dimension = dimension

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        logger.info(
            "Initialized OpenRouter embedding model: %s | dimension=%s",
            self.model_id,
            self.dimension,
        )

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:

        logger.info(
            "Creating embedding with OpenRouter model=%s",
            self.model_id,
        )

        try:
            response = await self.client.embeddings.create(
                model=self.model_id,
                input=text,
            )

            embedding = response.data[0].embedding

            logger.info(
                "Embedding created: model=%s | dimension=%s",
                self.model_id,
                len(embedding),
            )

            return embedding

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

        try:
            response = await self.client.embeddings.create(
                model=self.model_id,
                input=texts,
            )

            embeddings = [item.embedding for item in response.data]

            logger.info(
                "Document embeddings completed: model=%s | documents=%s",
                self.model_id,
                len(texts),
            )

            return embeddings

        except Exception:
            logger.exception(
                "OpenRouter document embeddings failed: model=%s",
                self.model_id,
            )
            raise

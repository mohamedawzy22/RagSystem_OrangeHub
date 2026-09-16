import logging

from ollama import AsyncClient

from ..chat_interface import ChatModel
from ..embedding_interface import EmbeddingModel

logger = logging.getLogger("uvicorn")


class OllamaChatModel(ChatModel):
    def __init__(
        self,
        model_id: str,
        base_url: str,
    ):
        self.model_id = model_id
        self.client = AsyncClient(host=base_url)

        logger.info(
            "Initialized Ollama chat model: %s",
            self.model_id,
        )

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

        options = {
            "temperature": temperature,
        }

        if max_tokens is not None:
            options["num_predict"] = max_tokens

        try:
            response = await self.client.chat(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                options=options,
            )

            logger.info(
                "Ollama generation completed: model=%s",
                self.model_id,
            )

            return response["message"]["content"]

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

        if max_tokens is not None:
            options["num_predict"] = max_tokens

        try:
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
            )

            async for chunk in response:
                content = chunk["message"]["content"]

                if content:
                    yield content

            logger.info(
                "Ollama stream completed: model=%s",
                self.model_id,
            )

        except Exception:
            logger.exception(
                "Ollama stream failed: model=%s",
                self.model_id,
            )
            raise


class OllamaEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_id: str,
        base_url: str,
        dimension: int,
    ):
        self.model_id = model_id
        self.dimension = dimension
        self.client = AsyncClient(host=base_url)

        logger.info(
            "Initialized Ollama embedding model: %s | dimension=%s",
            self.model_id,
            self.dimension,
        )

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:

        logger.info(
            "Creating embedding with Ollama model=%s",
            self.model_id,
        )

        try:
            response = await self.client.embed(
                model=self.model_id,
                input=text,
            )

            embedding = response["embeddings"][0]

            logger.info(
                "Embedding created: model=%s | dimension=%s",
                self.model_id,
                len(embedding),
            )

            return embedding

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

        try:
            response = await self.client.embed(
                model=self.model_id,
                input=texts,
            )

            embeddings = response["embeddings"]

            logger.info(
                "Document embeddings completed: model=%s | documents=%s",
                self.model_id,
                len(embeddings),
            )

            return embeddings

        except Exception:
            logger.exception(
                "Ollama document embeddings failed: model=%s",
                self.model_id,
            )
            raise

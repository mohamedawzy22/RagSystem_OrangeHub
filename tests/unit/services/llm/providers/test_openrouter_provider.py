from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.llm.providers.openrouter import (
    OpenRouterChatModel,
    OpenRouterEmbeddingModel,
)


@pytest.mark.anyio
async def test_openrouter_chat_generate():
    model = OpenRouterChatModel(
        model_id="test-model",
        api_key="test-key",
    )

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Hello from OpenRouter",
                )
            )
        ]
    )

    model.client = MagicMock()
    model.client.chat.completions.create = AsyncMock(return_value=response)

    result = await model.generate(
        prompt="Say hello",
        temperature=0.5,
        max_tokens=100,
    )

    assert result == "Hello from OpenRouter"

    model.client.chat.completions.create.assert_awaited_once_with(
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": "Say hello",
            }
        ],
        temperature=0.5,
        max_tokens=100,
    )


@pytest.mark.anyio
async def test_openrouter_chat_generate_without_max_tokens():
    model = OpenRouterChatModel(
        model_id="test-model",
        api_key="test-key",
    )

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Hello",
                )
            )
        ]
    )

    model.client = MagicMock()
    model.client.chat.completions.create = AsyncMock(return_value=response)

    result = await model.generate(
        prompt="Hello",
        temperature=0.7,
    )

    assert result == "Hello"

    model.client.chat.completions.create.assert_awaited_once_with(
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        temperature=0.7,
        max_tokens=None,
    )


@pytest.mark.anyio
async def test_openrouter_chat_generate_error():
    model = OpenRouterChatModel(
        model_id="test-model",
        api_key="test-key",
    )

    model.client = MagicMock()
    model.client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("OpenRouter error")
    )

    with pytest.raises(RuntimeError, match="OpenRouter error"):
        await model.generate("Hello")


@pytest.mark.anyio
async def test_openrouter_chat_stream():
    model = OpenRouterChatModel(
        model_id="test-model",
        api_key="test-key",
    )

    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=" from"))]
        ),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=""))]),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=" OpenRouter"))]
        ),
    ]

    async def mock_response():
        for chunk in chunks:
            yield chunk

    model.client = MagicMock()
    model.client.chat.completions.create = AsyncMock(return_value=mock_response())

    result = []

    async for chunk in model.stream(
        prompt="Say hello",
        temperature=0.5,
        max_tokens=50,
    ):
        result.append(chunk)

    assert result == [
        "Hello",
        " from",
        " OpenRouter",
    ]

    model.client.chat.completions.create.assert_awaited_once_with(
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": "Say hello",
            }
        ],
        temperature=0.5,
        max_tokens=50,
        stream=True,
    )


@pytest.mark.anyio
async def test_openrouter_chat_stream_error():
    model = OpenRouterChatModel(
        model_id="test-model",
        api_key="test-key",
    )

    model.client = MagicMock()
    model.client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("Stream error")
    )

    with pytest.raises(RuntimeError, match="Stream error"):
        async for _ in model.stream("Hello"):
            pass


@pytest.mark.anyio
async def test_openrouter_embedding_text():
    model = OpenRouterEmbeddingModel(
        model_id="test-embedding-model",
        api_key="test-key",
        dimension=1024,
    )

    response = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])

    model.client = MagicMock()
    model.client.embeddings.create = AsyncMock(return_value=response)

    result = await model.embed_text("Hello world")

    assert result == [0.1, 0.2, 0.3]

    model.client.embeddings.create.assert_awaited_once_with(
        model="test-embedding-model",
        input="Hello world",
    )


@pytest.mark.anyio
async def test_openrouter_embedding_text_error():
    model = OpenRouterEmbeddingModel(
        model_id="test-embedding-model",
        api_key="test-key",
        dimension=1024,
    )

    model.client = MagicMock()
    model.client.embeddings.create = AsyncMock(
        side_effect=RuntimeError("Embedding error")
    )

    with pytest.raises(RuntimeError, match="Embedding error"):
        await model.embed_text("Hello world")


@pytest.mark.anyio
async def test_openrouter_embedding_documents():
    model = OpenRouterEmbeddingModel(
        model_id="test-embedding-model",
        api_key="test-key",
        dimension=1024,
    )

    response = SimpleNamespace(
        data=[
            SimpleNamespace(embedding=[0.1, 0.2]),
            SimpleNamespace(embedding=[0.3, 0.4]),
        ]
    )

    model.client = MagicMock()
    model.client.embeddings.create = AsyncMock(return_value=response)

    result = await model.embed_documents(["Hello", "World"])

    assert result == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    model.client.embeddings.create.assert_awaited_once_with(
        model="test-embedding-model",
        input=["Hello", "World"],
    )


@pytest.mark.anyio
async def test_openrouter_embedding_documents_error():
    model = OpenRouterEmbeddingModel(
        model_id="test-embedding-model",
        api_key="test-key",
        dimension=1024,
    )

    model.client = MagicMock()
    model.client.embeddings.create = AsyncMock(
        side_effect=RuntimeError("Documents embedding error")
    )

    with pytest.raises(
        RuntimeError,
        match="Documents embedding error",
    ):
        await model.embed_documents(["Hello", "World"])

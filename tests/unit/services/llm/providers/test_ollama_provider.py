from unittest.mock import AsyncMock, MagicMock

import pytest

from services.llm.providers.ollama import (
    OllamaChatModel,
    OllamaEmbeddingModel,
)


@pytest.mark.anyio
async def test_ollama_chat_generate():
    model = OllamaChatModel(
        model_id="qwen3:8b",
        base_url="http://localhost:11434",
    )

    model.client = MagicMock()
    model.client.chat = AsyncMock(
        return_value={
            "message": {
                "content": "Hello from Ollama",
            }
        }
    )

    result = await model.generate(
        prompt="Say hello",
        temperature=0.5,
        max_tokens=100,
    )

    assert result == "Hello from Ollama"

    model.client.chat.assert_awaited_once_with(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": "Say hello",
            }
        ],
        options={
            "temperature": 0.5,
            "num_predict": 100,
        },
    )


@pytest.mark.anyio
async def test_ollama_chat_generate_without_max_tokens():
    model = OllamaChatModel(
        model_id="qwen3:8b",
        base_url="http://localhost:11434",
    )

    model.client = MagicMock()
    model.client.chat = AsyncMock(
        return_value={
            "message": {
                "content": "Hello",
            }
        }
    )

    result = await model.generate(
        prompt="Hello",
        temperature=0.7,
    )

    assert result == "Hello"

    model.client.chat.assert_awaited_once_with(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        options={
            "temperature": 0.7,
        },
    )


@pytest.mark.anyio
async def test_ollama_chat_generate_error():
    model = OllamaChatModel(
        model_id="qwen3:8b",
        base_url="http://localhost:11434",
    )

    model.client = MagicMock()
    model.client.chat = AsyncMock(side_effect=RuntimeError("Ollama error"))

    with pytest.raises(RuntimeError, match="Ollama error"):
        await model.generate("Hello")


@pytest.mark.anyio
async def test_ollama_chat_stream():
    model = OllamaChatModel(
        model_id="qwen3:8b",
        base_url="http://localhost:11434",
    )

    chunks = [
        {"message": {"content": "Hello"}},
        {"message": {"content": " from"}},
        {"message": {"content": ""}},
        {"message": {"content": " Ollama"}},
    ]

    async def mock_response():
        for chunk in chunks:
            yield chunk

    model.client = MagicMock()
    model.client.chat = AsyncMock(return_value=mock_response())

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
        " Ollama",
    ]

    model.client.chat.assert_awaited_once_with(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": "Say hello",
            }
        ],
        options={
            "temperature": 0.5,
            "num_predict": 50,
        },
        stream=True,
    )


@pytest.mark.anyio
async def test_ollama_chat_stream_error():
    model = OllamaChatModel(
        model_id="qwen3:8b",
        base_url="http://localhost:11434",
    )

    model.client = MagicMock()
    model.client.chat = AsyncMock(side_effect=RuntimeError("Stream error"))

    with pytest.raises(RuntimeError, match="Stream error"):
        async for _ in model.stream("Hello"):
            pass


@pytest.mark.anyio
async def test_ollama_embedding_text():
    model = OllamaEmbeddingModel(
        model_id="bge-m3",
        base_url="http://localhost:11434",
        dimension=1024,
    )

    model.client = MagicMock()
    model.client.embed = AsyncMock(
        return_value={
            "embeddings": [
                [0.1, 0.2, 0.3],
            ]
        }
    )

    result = await model.embed_text("Hello world")

    assert result == [0.1, 0.2, 0.3]

    model.client.embed.assert_awaited_once_with(
        model="bge-m3",
        input="Hello world",
    )


@pytest.mark.anyio
async def test_ollama_embedding_text_error():
    model = OllamaEmbeddingModel(
        model_id="bge-m3",
        base_url="http://localhost:11434",
        dimension=1024,
    )

    model.client = MagicMock()
    model.client.embed = AsyncMock(side_effect=RuntimeError("Embedding error"))

    with pytest.raises(RuntimeError, match="Embedding error"):
        await model.embed_text("Hello world")


@pytest.mark.anyio
async def test_ollama_embedding_documents():
    model = OllamaEmbeddingModel(
        model_id="bge-m3",
        base_url="http://localhost:11434",
        dimension=1024,
    )

    embeddings = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    model.client = MagicMock()
    model.client.embed = AsyncMock(
        return_value={
            "embeddings": embeddings,
        }
    )

    result = await model.embed_documents(["Hello", "World"])

    assert result == embeddings

    model.client.embed.assert_awaited_once_with(
        model="bge-m3",
        input=["Hello", "World"],
    )


@pytest.mark.anyio
async def test_ollama_embedding_documents_error():
    model = OllamaEmbeddingModel(
        model_id="bge-m3",
        base_url="http://localhost:11434",
        dimension=1024,
    )

    model.client = MagicMock()
    model.client.embed = AsyncMock(
        side_effect=RuntimeError("Documents embedding error")
    )

    with pytest.raises(
        RuntimeError,
        match="Documents embedding error",
    ):
        await model.embed_documents(["Hello", "World"])

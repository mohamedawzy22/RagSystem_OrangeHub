from unittest.mock import AsyncMock, MagicMock

import pytest

from services.llm.fallback_chat import FallbackChatModel


def create_fallback_model():
    primary = MagicMock()
    fallback = MagicMock()

    model = FallbackChatModel(
        primary=primary,
        fallback=fallback,
        primary_name="qwen3",
        fallback_name="llama3.2",
    )

    return model, primary, fallback


@pytest.mark.anyio
async def test_generate_primary_success():
    model, primary, fallback = create_fallback_model()

    primary.generate = AsyncMock(
        return_value="Primary response",
    )

    fallback.generate = AsyncMock(
        return_value="Fallback response",
    )

    result = await model.generate(
        prompt="Hello",
        temperature=0.5,
        max_tokens=100,
    )

    assert result == "Primary response"

    primary.generate.assert_awaited_once_with(
        prompt="Hello",
        temperature=0.5,
        max_tokens=100,
    )

    fallback.generate.assert_not_awaited()


@pytest.mark.anyio
async def test_generate_uses_fallback_when_primary_fails():
    model, primary, fallback = create_fallback_model()

    primary.generate = AsyncMock(
        side_effect=RuntimeError("Primary failed"),
    )

    fallback.generate = AsyncMock(
        return_value="Fallback response",
    )

    result = await model.generate(
        prompt="Hello",
        temperature=0.7,
        max_tokens=100,
    )

    assert result == "Fallback response"

    primary.generate.assert_awaited_once_with(
        prompt="Hello",
        temperature=0.7,
        max_tokens=100,
    )

    fallback.generate.assert_awaited_once_with(
        prompt="Hello",
        temperature=0.7,
        max_tokens=100,
    )


@pytest.mark.anyio
async def test_generate_raises_when_both_models_fail():
    model, primary, fallback = create_fallback_model()

    primary.generate = AsyncMock(
        side_effect=RuntimeError("Primary failed"),
    )

    fallback.generate = AsyncMock(
        side_effect=RuntimeError("Fallback failed"),
    )

    with pytest.raises(
        RuntimeError,
        match="Fallback failed",
    ):
        await model.generate(
            prompt="Hello",
        )

    primary.generate.assert_awaited_once()
    fallback.generate.assert_awaited_once()


async def async_chunks(chunks):
    for chunk in chunks:
        yield chunk


@pytest.mark.anyio
async def test_stream_primary_success():
    model, primary, fallback = create_fallback_model()

    primary.stream = MagicMock(
        return_value=async_chunks(
            [
                "Hello",
                " from",
                " primary",
            ]
        )
    )

    fallback.stream = MagicMock(
        return_value=async_chunks(
            [
                "Fallback",
            ]
        )
    )

    result = []

    async for chunk in model.stream(
        prompt="Hello",
        temperature=0.5,
        max_tokens=50,
    ):
        result.append(chunk)

    assert result == [
        "Hello",
        " from",
        " primary",
    ]

    primary.stream.assert_called_once_with(
        prompt="Hello",
        temperature=0.5,
        max_tokens=50,
    )

    fallback.stream.assert_not_called()


@pytest.mark.anyio
async def test_stream_uses_fallback_when_primary_fails_before_response():
    model, primary, fallback = create_fallback_model()

    async def primary_stream():
        raise RuntimeError("Primary stream failed")
        yield

    primary.stream = MagicMock(return_value=primary_stream())

    fallback.stream = MagicMock(
        return_value=async_chunks(
            [
                "Fallback",
                " response",
            ]
        )
    )

    result = []

    async for chunk in model.stream(
        prompt="Hello",
    ):
        result.append(chunk)

    assert result == [
        "Fallback",
        " response",
    ]

    primary.stream.assert_called_once_with(
        prompt="Hello",
        temperature=0.7,
        max_tokens=None,
    )

    fallback.stream.assert_called_once_with(
        prompt="Hello",
        temperature=0.7,
        max_tokens=None,
    )


@pytest.mark.anyio
async def test_stream_does_not_fallback_after_partial_response():
    model, primary, fallback = create_fallback_model()

    async def primary_stream():
        yield "Partial"
        raise RuntimeError("Primary stream failed")

    primary.stream = MagicMock(return_value=primary_stream())

    fallback.stream = MagicMock(
        return_value=async_chunks(
            [
                "Fallback",
            ]
        )
    )

    result = []

    with pytest.raises(
        RuntimeError,
        match="Primary stream failed",
    ):
        async for chunk in model.stream(
            prompt="Hello",
        ):
            result.append(chunk)

    assert result == ["Partial"]

    primary.stream.assert_called_once()

    fallback.stream.assert_not_called()


@pytest.mark.anyio
async def test_stream_raises_when_fallback_fails():
    model, primary, fallback = create_fallback_model()

    async def primary_stream():
        raise RuntimeError("Primary failed")
        yield

    primary.stream = MagicMock(return_value=primary_stream())

    async def fallback_stream():
        raise RuntimeError("Fallback failed")
        yield

    fallback.stream = MagicMock(return_value=fallback_stream())

    with pytest.raises(
        RuntimeError,
        match="Fallback failed",
    ):
        async for _ in model.stream(
            prompt="Hello",
        ):
            pass

    primary.stream.assert_called_once()
    fallback.stream.assert_called_once()


def test_generate_skips_failed_primary_during_cooldown():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from services.llm.fallback_chat import FallbackChatModel

    primary = MagicMock()
    primary.generate = AsyncMock(side_effect=RuntimeError("primary unavailable"))

    fallback = MagicMock()
    fallback.generate = AsyncMock(return_value="fallback response")

    model = FallbackChatModel(
        primary=primary,
        fallback=fallback,
        primary_name="primary",
        fallback_name="fallback",
        cooldown_seconds=60,
    )

    async def run():
        first = await model.generate("test")
        second = await model.generate("test")

        assert first == "fallback response"
        assert second == "fallback response"

    asyncio.run(run())

    assert primary.generate.await_count == 1
    assert fallback.generate.await_count == 2


def test_generate_retries_primary_after_cooldown():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from services.llm.fallback_chat import FallbackChatModel

    primary = MagicMock()
    primary.generate = AsyncMock(
        side_effect=[
            RuntimeError("primary unavailable"),
            "primary recovered",
        ]
    )

    fallback = MagicMock()
    fallback.generate = AsyncMock(return_value="fallback response")

    model = FallbackChatModel(
        primary=primary,
        fallback=fallback,
        primary_name="primary",
        fallback_name="fallback",
        cooldown_seconds=0,
    )

    async def run():
        first = await model.generate("test")
        second = await model.generate("test")

        assert first == "fallback response"
        assert second == "primary recovered"

    asyncio.run(run())

    assert primary.generate.await_count == 2
    assert fallback.generate.await_count == 1

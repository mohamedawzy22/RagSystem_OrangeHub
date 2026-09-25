import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from services.llm.fallback_chat import FallbackChatModel
from services.llm.llm_manager import ModelManager


def test_fallback_primary_unavailable():
    primary = MagicMock()
    primary.generate = AsyncMock()
    primary.health_check = AsyncMock(return_value=False)

    fallback = MagicMock()
    fallback.generate = AsyncMock(return_value="fallback")
    fallback.health_check = AsyncMock(return_value=True)

    model = FallbackChatModel(
        primary=primary,
        fallback=fallback,
        primary_name="primary",
        fallback_name="fallback",
        cooldown_seconds=60,
    )

    model.set_primary_unavailable()

    result = asyncio.run(model.generate("test"))

    assert result == "fallback"

    primary.generate.assert_not_awaited()
    fallback.generate.assert_awaited_once()


def test_model_manager_primary_failure_uses_fallback():
    primary = MagicMock()
    primary.generate = AsyncMock(return_value="primary response")
    primary.health_check = AsyncMock(return_value=False)

    fallback = MagicMock()
    fallback.generate = AsyncMock(return_value="fallback response")
    fallback.health_check = AsyncMock(return_value=True)

    embedding = MagicMock()
    embedding.health_check = AsyncMock(return_value=True)

    settings = SimpleNamespace(
        CHAT_PRIMARY_MODEL="primary",
        CHAT_FALLBACK_MODEL="fallback",
        EMBEDDING_MODEL="embedding",
        CHAT_FALLBACK_COOLDOWN_SECONDS=60,
    )

    factory = MagicMock()
    factory.settings = settings

    manager = ModelManager(factory)

    manager._chat_models = {
        "primary": primary,
        "fallback": fallback,
    }

    manager._embedding_models = {
        "embedding": embedding,
    }

    manager._selected_chat_model = FallbackChatModel(
        primary=primary,
        fallback=fallback,
        primary_name="primary",
        fallback_name="fallback",
        cooldown_seconds=60,
    )

    manager._selected_embedding_model = embedding

    asyncio.run(manager.health_check_models())

    result = asyncio.run(manager.get_selected_chat_model().generate("test"))

    assert result == "fallback response"

    primary.health_check.assert_awaited_once()
    fallback.health_check.assert_awaited_once()
    embedding.health_check.assert_awaited_once()

    primary.generate.assert_not_awaited()
    fallback.generate.assert_awaited_once()


def test_fallback_health_check_handles_primary_exception():
    primary = MagicMock()
    primary.health_check = AsyncMock(side_effect=RuntimeError("primary unavailable"))

    fallback = MagicMock()
    fallback.health_check = AsyncMock(return_value=True)

    model = FallbackChatModel(
        primary=primary,
        fallback=fallback,
        primary_name="primary",
        fallback_name="fallback",
        cooldown_seconds=60,
    )

    result = asyncio.run(model.health_check())

    assert result is True
    primary.health_check.assert_awaited_once()
    fallback.health_check.assert_awaited_once()

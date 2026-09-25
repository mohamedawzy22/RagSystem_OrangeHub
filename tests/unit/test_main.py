import asyncio
from unittest.mock import AsyncMock, MagicMock


def test_lifespan_initializes_and_closes_resources(monkeypatch):
    import main

    settings = MagicMock()
    settings.MONGODB_URL = "mongodb://test"
    settings.MONGODB_DATABASE = "test-db"

    monkeypatch.setattr(main, "get_setting", lambda: settings)

    mongo_client = MagicMock()
    db_client = MagicMock()
    mongo_client.__getitem__.return_value = db_client

    monkeypatch.setattr(
        main,
        "AsyncIOMotorClient",
        MagicMock(return_value=mongo_client),
    )

    model_factory = MagicMock()
    model_manager = MagicMock()
    model_manager.get_selected_chat_model.return_value = MagicMock()
    model_manager.get_selected_embedding_model.return_value = MagicMock()

    monkeypatch.setattr(
        main,
        "ModelFactory",
        MagicMock(return_value=model_factory),
    )
    monkeypatch.setattr(
        main,
        "ModelManager",
        MagicMock(return_value=model_manager),
    )

    vector_db_factory = MagicMock()
    vector_db_manager = MagicMock()
    vector_db = MagicMock()
    vector_db.create_collection = AsyncMock()

    vector_db_manager.get_database.return_value = vector_db

    monkeypatch.setattr(
        main,
        "VectorDBFactory",
        MagicMock(return_value=vector_db_factory),
    )
    monkeypatch.setattr(
        main,
        "VectorDBManager",
        MagicMock(return_value=vector_db_manager),
    )

    rag_controller = MagicMock()

    monkeypatch.setattr(
        main,
        "RAGController",
        MagicMock(return_value=rag_controller),
    )

    app = MagicMock()

    async def run_lifespan():
        async with main.lifespan(app):
            assert app.mongo_conn is mongo_client
            assert app.db_client is db_client
            assert app.model_manager is model_manager
            assert app.vector_db_manager is vector_db_manager
            assert app.rag_controller is rag_controller

    asyncio.run(run_lifespan())

    mongo_client.close.assert_called_once()
    model_manager.load_models.assert_called_once()
    vector_db_manager.load_databases.assert_called_once()
    vector_db_manager.get_database.assert_called_once_with("default")
    vector_db.create_collection.assert_awaited_once()

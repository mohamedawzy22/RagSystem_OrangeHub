from contextlib import asynccontextmanager

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient

from controllers.rag_controller import RAGController
from helpers.config import get_setting
from routes import base, data, rag
from services.llm.llm_factory import ModelFactory
from services.llm.llm_manager import ModelManager
from services.vectordb.vectordb_factory import VectorDBFactory
from services.vectordb.vectordb_manager import VectorDBManager
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_setting()

    setup_logging(settings.LOG_LEVEL)

    app.mongo_conn = None
    app.model_manager = None
    app.vector_db_manager = None

    logger.info(
        "Starting %s version=%s",
        settings.APP_NAME,
        settings.APP_VERSION,
    )

    try:
        # -------------------------
        # MongoDB
        # -------------------------

        app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)

        app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]

        logger.info("MongoDB connection initialized")

        # -------------------------
        # LLM
        # -------------------------

        model_factory = ModelFactory(settings)
        model_manager = ModelManager(model_factory)

        model_manager.load_models()

        await model_manager.health_check_models()

        if settings.MODEL_WARMUP_ENABLED:
            await model_manager.warm_up_models(
                timeout_seconds=settings.MODEL_WARMUP_TIMEOUT_SECONDS,
            )

        app.model_manager = model_manager

        logger.info("LLM models initialized")

        # -------------------------
        # Vector Database
        # -------------------------

        vector_db_factory = VectorDBFactory(settings)
        vector_db_manager = VectorDBManager(vector_db_factory)

        vector_db_manager.load_databases()

        vector_db = vector_db_manager.get_database("default")

        await vector_db.create_collection()

        app.vector_db_manager = vector_db_manager

        logger.info("Vector database initialized")

        # -------------------------
        # RAG Controller
        # -------------------------

        app.rag_controller = RAGController(
            chat_model=model_manager.get_selected_chat_model(),
            embedding_model=model_manager.get_selected_embedding_model(),
            vector_db_manager=vector_db_manager,
            db_client=app.db_client,
        )

        logger.info("RAG controller initialized")

        yield

    finally:
        logger.info("Application shutdown started")

        # -------------------------
        # Vector Database
        # -------------------------

        if app.vector_db_manager is not None:
            await app.vector_db_manager.close()

        # -------------------------
        # LLM
        # -------------------------

        if app.model_manager is not None:
            await app.model_manager.close()

        # -------------------------
        # MongoDB
        # -------------------------

        if app.mongo_conn is not None:
            app.mongo_conn.close()

            logger.info("MongoDB connection closed")

        logger.info("Application shutdown completed")


app = FastAPI(lifespan=lifespan)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(rag.rag_router)

import logging
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

logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):

    settings = get_setting()

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
        embedding_model=(model_manager.get_selected_embedding_model()),
        vector_db_manager=vector_db_manager,
        db_client=app.db_client,
    )

    logger.info("RAG controller initialized")

    yield

    # -------------------------
    # Shutdown
    # -------------------------

    app.mongo_conn.close()

    logger.info("MongoDB connection closed")


app = FastAPI(lifespan=lifespan)


app.include_router(base.base_router)

app.include_router(data.data_router)

app.include_router(rag.rag_router)

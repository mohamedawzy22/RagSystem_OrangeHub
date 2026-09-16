import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient

from helpers.config import get_setting
from routes import base, chat, data
from services.llm.factory import ModelFactory
from services.llm.manager import ModelManager

logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_setting()

    # Initialize MongoDB
    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]

    logger.info("MongoDB connection initialized")

    # Initialize LLM models
    model_factory = ModelFactory(settings)
    model_manager = ModelManager(model_factory)

    model_manager.load_models()

    app.model_manager = model_manager

    logger.info("LLM models initialized")

    yield

    # Close MongoDB connection
    app.mongo_conn.close()

    logger.info("MongoDB connection closed")


app = FastAPI(lifespan=lifespan)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(chat.chat_router)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient

from helpers.config import get_setting
from routes import base, data


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_setting()

    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)

    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]

    yield

    app.mongo_conn.close()


app = FastAPI(lifespan=lifespan)


app.include_router(base.base_router)
app.include_router(data.data_router)

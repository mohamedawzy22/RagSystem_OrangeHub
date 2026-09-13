from fastapi import APIRouter, Depends
from helpers.config import get_setting, Setting

base_router = APIRouter(prefix="/api/v1", tags=["api_v1"])


@base_router.get("/health")
async def health_check(app_setting: Setting = Depends(get_setting)):
    api_name = app_setting.APP_NAME
    api_version = app_setting.APP_VERSION

    return {"API_Name": api_name, "API_Version": api_version}

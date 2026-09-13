from fastapi import APIRouter, Depends, UploadFile, status
from fastapi.responses import JSONResponse
from helpers.config import get_setting, Setting
from controllers import DataController
from models import ResponseSignal
import aiofiles
import logging


logger = logging.getLogger("uvicorn.error")

data_router = APIRouter(prefix="/api/v1/data", tags=["api_v1", "data"])


@data_router.post("/upload/{project_id}")
async def upload_data(
    project_id: int, file: UploadFile, app_setting: Setting = Depends(get_setting)
):
    data_controller = DataController()

    # 1. Validate file
    is_valid, result = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, content={"signal": result}
        )

    # 2. Generate unique file path
    file_dir, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename, project_id=project_id
    )

    # 3. Save file
    try:
        async with aiofiles.open(file_dir, "wb") as f:
            while chunk := await file.read(app_setting.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)

    except Exception as e:
        logger.error(f"Error occurred while uploading file: {e}")

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.FILE_UPLOAD_FAILED.value},
        )

    # 4. Return success
    return JSONResponse(
        content={
            "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
            "file_name": file_id,
            "file_path": file_dir,
        }
    )

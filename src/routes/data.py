from fastapi import APIRouter, Depends, UploadFile

from core.dependencies import (
    get_processing_service,
    get_upload_service,
)
from services.processing.processing_service import ProcessingService
from services.storage.upload_service import UploadService

from .schemes.data import ProcessRequest

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"],
)


@data_router.post("/upload/{project_id}")
async def upload_data(
    project_id: str,
    file: UploadFile,
    upload_service: UploadService = Depends(
        get_upload_service,
    ),
):
    return await upload_service.upload(
        project_id=project_id,
        file=file,
    )


@data_router.post("/process/{project_id}")
async def process_endpoint(
    project_id: str,
    process_request: ProcessRequest,
    processing_service: ProcessingService = Depends(
        get_processing_service,
    ),
):
    result = await processing_service.process(
        project_id=project_id,
        chunk_size=process_request.chunk_size,
        overlap_size=process_request.overlap_size,
        do_reset=process_request.do_reset,
        file_id=process_request.file_id,
    )

    return result

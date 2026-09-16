import logging
import os

import aiofiles
from fastapi import APIRouter, Depends, Request, UploadFile, status
from fastapi.responses import JSONResponse

from controllers import DataController, ProcessController
from helpers.config import Setting, get_setting
from models import ResponseSignal
from models.asset_model import AssetModel
from models.chunk_model import ChunkModel
from models.db_schemes import Asset, DataChunk
from models.enums.asset_type_enum import AssetTypeEnum
from models.project_model import ProjectModel

from .schemes.data import ProcessRequest

logger = logging.getLogger("uvicorn.error")


data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"],
)


@data_router.post("/upload/{project_id}")
async def upload_data(
    request: Request,
    project_id: str,
    file: UploadFile,
    app_settings: Setting = Depends(get_setting),
):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)

    project = await project_model.get_project_or_create_one(project_id=project_id)

    # Validate uploaded file.
    data_controller = DataController()

    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": result_signal,
            },
        )

    # Generate unique file path and file id.
    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id,
    )

    # Save file to disk.
    try:
        async with aiofiles.open(
            file_path,
            "wb",
        ) as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)

    except Exception:
        logger.exception(
            "Error while uploading file: file_id=%s",
            file_id,
        )

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value,
            },
        )

    # Store asset in database.
    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)

    asset_resource = Asset(
        asset_project_id=project.id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_name=file_id,
        asset_size=os.path.getsize(file_path),
    )

    try:
        asset_record = await asset_model.create_asset(asset=asset_resource)

    except Exception:
        logger.exception(
            "Error while saving asset: file_id=%s",
            file_id,
        )

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value,
            },
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
            "asset_name": asset_record.asset_name,
        },
    )


@data_router.post("/process/{project_id}")
async def process_endpoint(
    request: Request,
    project_id: str,
    process_request: ProcessRequest,
):
    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size
    do_reset = process_request.do_reset

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)

    project = await project_model.get_project_or_create_one(project_id=project_id)

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)

    # Get files to process.
    project_files_ids = {}

    if process_request.file_id:
        asset_record = await asset_model.get_asset_record(
            asset_project_id=project.id,
            asset_name=process_request.file_id,
        )

        if asset_record is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.FILE_ID_ERROR.value,
                },
            )

        project_files_ids = {
            asset_record.id: asset_record.asset_name,
        }

    else:
        project_files = await asset_model.get_all_project_assets(
            asset_project_id=project.id,
            asset_type=AssetTypeEnum.FILE.value,
        )

        project_files_ids = {record.id: record.asset_name for record in project_files}

    if not project_files_ids:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.NO_FILES_ERROR.value,
            },
        )

    # Create process controller with chunk configuration.
    process_controller = ProcessController(
        project_id=project_id,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
    )

    chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)

    # Delete existing chunks if requested.
    if do_reset == 1:
        await chunk_model.delete_chunks_by_project_id(project_id=project.id)

    inserted_chunks = 0
    processed_files = 0

    for asset_id, file_id in project_files_ids.items():
        logger.info(
            "Processing file: file_id=%s",
            file_id,
        )

        # Load file.
        file_content = process_controller.get_file_content(file_id=file_id)

        if file_content is None:
            logger.error(
                "Failed to load file: file_id=%s",
                file_id,
            )

            continue

        # Create chunks.
        file_chunks = process_controller.process_file_content(
            file_content=file_content,
            file_id=file_id,
        )

        if not file_chunks:
            logger.warning(
                "No chunks generated: file_id=%s",
                file_id,
            )

            continue

        # Convert chunks to database records.
        file_chunks_records = []

        for chunk in file_chunks:
            chunk_text = chunk.page_content.strip()

            if not chunk_text:
                logger.warning(
                    "Skipping empty chunk: file_id=%s",
                    file_id,
                )

                continue

            file_chunks_records.append(
                DataChunk(
                    chunk_text=chunk_text,
                    chunk_metadata=chunk.metadata,
                    chunk_order=len(file_chunks_records) + 1,
                    chunk_project_id=project.id,
                    chunk_asset_id=asset_id,
                )
            )

        if not file_chunks_records:
            logger.warning(
                "No valid chunks to insert: file_id=%s",
                file_id,
            )

            continue

        # Insert chunks into database.
        try:
            inserted_count = await chunk_model.insert_many_chunks(
                chunks=file_chunks_records
            )

            inserted_chunks += inserted_count
            processed_files += 1

            logger.info(
                "File processed successfully: file_id=%s, chunks=%s",
                file_id,
                inserted_count,
            )

        except Exception:
            logger.exception(
                "Failed to insert chunks: file_id=%s",
                file_id,
            )

            continue

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseSignal.PROCESSING_SUCCESS.value,
            "inserted_chunks": inserted_chunks,
            "processed_files": processed_files,
        },
    )

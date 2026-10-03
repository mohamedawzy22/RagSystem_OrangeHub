import os

import aiofiles
from fastapi import UploadFile

from core.exceptions import BadRequestError
from models import ResponseSignal
from models.asset_model import AssetModel
from models.db_schemes import Asset
from models.enums.asset_type_enum import AssetTypeEnum
from models.project_model import ProjectModel
from services.storage.project_storage import ProjectStorage
from utils.logger import get_logger

logger = get_logger(__name__)


class UploadService:
    def __init__(
        self,
        project_model: ProjectModel,
        asset_model: AssetModel,
        project_storage: ProjectStorage,
        chunk_size: int,
        max_file_size_mb: int,
        allowed_file_types: list[str],
    ):
        self.project_model = project_model
        self.asset_model = asset_model
        self.project_storage = project_storage
        self.chunk_size = chunk_size
        self.max_file_size = max_file_size_mb * 1048576
        self.allowed_file_types = set(allowed_file_types)

        logger.info(
            "Upload service initialized",
        )

    def _validate_file(
        self,
        file: UploadFile | None,
    ) -> None:
        if file is None:
            raise BadRequestError(
                ResponseSignal.FILE_NOT_FOUND.value,
            )

        if file.content_type not in self.allowed_file_types:
            raise BadRequestError(
                ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value,
            )

        if file.size is not None and file.size > self.max_file_size:
            raise BadRequestError(
                ResponseSignal.FILE_SIZE_EXCEEDED.value,
            )

        if not file.filename:
            raise BadRequestError(
                ResponseSignal.FILE_NOT_FOUND.value,
            )

    async def upload(
        self,
        project_id: str,
        file: UploadFile,
    ) -> dict:
        self._validate_file(file)

        project = await self.project_model.get_project_or_create_one(
            project_id=project_id,
        )

        file_name = self.project_storage.sanitize_file_name(
            file.filename,
        )

        file_path, generated_file_name = self.project_storage.generate_unique_filepath(
            project_id=project_id,
            file_name=file_name,
        )

        try:
            async with aiofiles.open(
                file_path,
                "wb",
            ) as output_file:
                while chunk := await file.read(
                    self.chunk_size,
                ):
                    await output_file.write(chunk)

        except Exception as exc:
            logger.exception(
                "Failed to save uploaded file: project_id=%s | file_id=%s",
                project_id,
                generated_file_name,
            )

            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                logger.exception(
                    "Failed to remove incomplete uploaded file: path=%s",
                    file_path,
                )

            raise BadRequestError(
                ResponseSignal.FILE_UPLOAD_FAILED.value,
            ) from exc

        asset = Asset(
            asset_project_id=project.id,
            asset_type=AssetTypeEnum.FILE.value,
            asset_name=generated_file_name,
            asset_size=os.path.getsize(file_path),
        )

        try:
            asset_record = await self.asset_model.create_asset(
                asset=asset,
            )

        except Exception as exc:
            logger.exception(
                "Failed to save asset record: project_id=%s | file_id=%s",
                project_id,
                generated_file_name,
            )

            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                logger.exception(
                    "Failed to remove uploaded file after DB failure: path=%s",
                    file_path,
                )

            raise BadRequestError(
                ResponseSignal.FILE_UPLOAD_FAILED.value,
            ) from exc

        logger.info(
            "File uploaded successfully: project_id=%s | file_id=%s",
            project_id,
            asset_record.asset_name,
        )

        return {
            "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
            "asset_name": asset_record.asset_name,
        }

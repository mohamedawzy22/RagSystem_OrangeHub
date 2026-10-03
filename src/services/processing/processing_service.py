import asyncio

from models import ResponseSignal
from models.asset_model import AssetModel
from models.chunk_model import ChunkModel
from models.db_schemes import DataChunk
from models.enums.asset_type_enum import AssetTypeEnum
from models.project_model import ProjectModel
from services.processing.document_processor import (
    DocumentProcessor,
)
from services.storage.project_storage import ProjectStorage
from utils.logger import get_logger

logger = get_logger(__name__)


class ProcessingService:
    def __init__(
        self,
        project_model: ProjectModel,
        asset_model: AssetModel,
        chunk_model: ChunkModel,
        project_storage: ProjectStorage,
    ):
        self.project_model = project_model
        self.asset_model = asset_model
        self.chunk_model = chunk_model
        self.project_storage = project_storage

        logger.info(
            "Processing service initialized successfully",
        )

    async def process(
        self,
        project_id: str,
        chunk_size: int = 100,
        overlap_size: int = 20,
        do_reset: int = 0,
        file_id: str | None = None,
    ) -> dict:
        logger.info(
            "Processing started: "
            "project_id=%s | chunk_size=%s | "
            "overlap=%s | do_reset=%s | file_id=%s",
            project_id,
            chunk_size,
            overlap_size,
            do_reset,
            file_id,
        )

        project = await self.project_model.get_project_or_create_one(
            project_id=project_id,
        )

        project_files_ids = {}

        if file_id:
            asset_record = await self.asset_model.get_asset_record(
                asset_project_id=project.id,
                asset_name=file_id,
            )

            if asset_record is None:
                logger.warning(
                    "Requested file not found: project_id=%s | file_id=%s",
                    project_id,
                    file_id,
                )

                return {
                    "signal": ResponseSignal.FILE_ID_ERROR.value,
                }

            project_files_ids = {
                asset_record.id: asset_record.asset_name,
            }

        else:
            project_files = await self.asset_model.get_all_project_assets(
                asset_project_id=project.id,
                asset_type=AssetTypeEnum.FILE.value,
            )

            project_files_ids = {
                record.id: record.asset_name for record in project_files
            }

        if not project_files_ids:
            logger.warning(
                "No files found for processing: project_id=%s",
                project_id,
            )

            return {
                "signal": ResponseSignal.NO_FILES_ERROR.value,
            }

        if do_reset == 1:
            await self.chunk_model.delete_chunks_by_project_id(
                project_id=project.id,
            )

            logger.info(
                "Existing chunks deleted: project_id=%s",
                project_id,
            )

        project_path = self.project_storage.get_project_path(
            project_id=project_id,
        )

        document_processor = DocumentProcessor(
            project_path=project_path,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
        )

        inserted_chunks = 0
        processed_files = 0

        for asset_id, current_file_id in project_files_ids.items():
            logger.info(
                "Processing file: file_id=%s",
                current_file_id,
            )

            file_content = await asyncio.to_thread(
                document_processor.get_file_content,
                file_id=current_file_id,
            )

            if file_content is None:
                logger.error(
                    "Failed to load file: file_id=%s",
                    current_file_id,
                )

                continue

            file_chunks = await asyncio.to_thread(
                document_processor.process_file_content,
                file_content=file_content,
                file_id=current_file_id,
            )

            if not file_chunks:
                logger.warning(
                    "No chunks generated: file_id=%s",
                    current_file_id,
                )

                continue

            file_chunks_records = []

            for chunk in file_chunks:
                chunk_text = chunk.page_content.strip()

                if not chunk_text:
                    logger.warning(
                        "Skipping empty chunk: file_id=%s",
                        current_file_id,
                    )
                    continue

                file_chunks_records.append(
                    DataChunk(
                        chunk_text=chunk_text,
                        chunk_metadata=chunk.metadata,
                        chunk_order=(len(file_chunks_records) + 1),
                        chunk_project_id=project.id,
                        chunk_asset_id=asset_id,
                    )
                )

            if not file_chunks_records:
                logger.warning(
                    "No valid chunks to insert: file_id=%s",
                    current_file_id,
                )

                continue

            try:
                inserted_count = await self.chunk_model.insert_many_chunks(
                    chunks=file_chunks_records,
                )

                inserted_chunks += inserted_count
                processed_files += 1

                logger.info(
                    "File processed successfully: file_id=%s | chunks=%s",
                    current_file_id,
                    inserted_count,
                )

            except Exception:
                logger.exception(
                    "Failed to insert chunks: file_id=%s",
                    current_file_id,
                )

                continue

        logger.info(
            "Processing completed: "
            "project_id=%s | processed_files=%s | "
            "inserted_chunks=%s",
            project_id,
            processed_files,
            inserted_chunks,
        )

        return {
            "signal": ResponseSignal.PROCESSING_SUCCESS.value,
            "inserted_chunks": inserted_chunks,
            "processed_files": processed_files,
        }

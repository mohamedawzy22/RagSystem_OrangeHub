from fastapi import Request

from controllers.rag_controller import RAGController
from models.asset_model import AssetModel
from models.chunk_model import ChunkModel
from models.project_model import ProjectModel
from services.processing.processing_service import ProcessingService
from services.storage.project_storage import ProjectStorage
from services.storage.upload_service import UploadService


def get_rag_controller(
    request: Request,
) -> RAGController:
    controller = request.app.state.container.rag_controller

    if controller is None:
        raise RuntimeError(
            "RAG controller is not initialized",
        )

    return controller


def get_project_model(
    request: Request,
) -> ProjectModel:
    project_model = request.app.state.container.project_model

    if project_model is None:
        raise RuntimeError(
            "Project model is not initialized",
        )

    return project_model


def get_asset_model(
    request: Request,
) -> AssetModel:
    asset_model = request.app.state.container.asset_model

    if asset_model is None:
        raise RuntimeError(
            "Asset model is not initialized",
        )

    return asset_model


def get_chunk_model(
    request: Request,
) -> ChunkModel:
    chunk_model = request.app.state.container.chunk_model

    if chunk_model is None:
        raise RuntimeError(
            "Chunk model is not initialized",
        )

    return chunk_model


def get_processing_service(
    request: Request,
) -> ProcessingService:
    processing_service = request.app.state.container.processing_service

    if processing_service is None:
        raise RuntimeError(
            "Processing service is not initialized",
        )

    return processing_service


def get_project_storage(
    request: Request,
) -> ProjectStorage:
    project_storage = request.app.state.container.project_storage

    if project_storage is None:
        raise RuntimeError(
            "Project storage is not initialized",
        )

    return project_storage


def get_upload_service(
    request: Request,
) -> UploadService:
    upload_service = request.app.state.container.upload_service

    if upload_service is None:
        raise RuntimeError(
            "Upload service is not initialized",
        )

    return upload_service

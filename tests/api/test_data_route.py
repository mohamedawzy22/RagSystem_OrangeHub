from types import SimpleNamespace
from unittest.mock import AsyncMock

from bson import ObjectId

from controllers import DataController, ProcessController
from models import ResponseSignal
from models.asset_model import AssetModel
from models.chunk_model import ChunkModel
from models.project_model import ProjectModel
from tests.factories import make_asset, make_project

PROJECT_ID = ObjectId("68c123456789abcdef123456")
ASSET_ID = ObjectId("68c987654321abcdef123456")


async def mock_project_create_instance(db_client):
    mock_model = SimpleNamespace()
    mock_model.get_project_or_create_one = AsyncMock(
        return_value=make_project(object_id=PROJECT_ID)
    )
    return mock_model


async def mock_asset_create_instance(db_client):
    mock_model = SimpleNamespace()
    mock_model.create_asset = AsyncMock(
        return_value=make_asset(
            object_id=ASSET_ID,
            asset_name="test.txt",
        )
    )
    return mock_model


def test_upload_invalid_file(client, monkeypatch):
    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        mock_project_create_instance,
    )

    def mock_validate(self, file):
        return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value

    monkeypatch.setattr(
        DataController,
        "validate_uploaded_file",
        mock_validate,
    )

    response = client.post(
        "/api/v1/data/upload/1",
        files={
            "file": (
                "test.exe",
                b"fake file",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["signal"] == ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value


def test_upload_success(tmp_path, client, monkeypatch):
    file_path = tmp_path / "test.txt"

    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        mock_project_create_instance,
    )

    monkeypatch.setattr(
        AssetModel,
        "create_instance",
        mock_asset_create_instance,
    )

    def mock_validate(self, file):
        return True, "FILE_VALIDATED_SUCCESS"

    def mock_generate(self, orig_file_name, project_id):
        return str(file_path), "test.txt"

    monkeypatch.setattr(
        DataController,
        "validate_uploaded_file",
        mock_validate,
    )

    monkeypatch.setattr(
        DataController,
        "generate_unique_filepath",
        mock_generate,
    )

    response = client.post(
        "/api/v1/data/upload/1",
        files={
            "file": (
                "test.txt",
                b"Hello World",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["asset_name"] == "test.txt"
    assert file_path.read_bytes() == b"Hello World"


def test_upload_failed_to_save(tmp_path, client, monkeypatch):
    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        mock_project_create_instance,
    )

    def mock_validate(self, file):
        return True, "FILE_VALIDATED_SUCCESS"

    def mock_generate(self, orig_file_name, project_id):
        return str(tmp_path), "test.txt"

    monkeypatch.setattr(
        DataController,
        "validate_uploaded_file",
        mock_validate,
    )

    monkeypatch.setattr(
        DataController,
        "generate_unique_filepath",
        mock_generate,
    )

    response = client.post(
        "/api/v1/data/upload/1",
        files={
            "file": (
                "test.txt",
                b"Hello World",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_process_file_not_found(client, monkeypatch):
    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        mock_project_create_instance,
    )

    mock_asset_model = SimpleNamespace()
    mock_asset_model.get_asset_record = AsyncMock(return_value=None)

    monkeypatch.setattr(
        AssetModel,
        "create_instance",
        AsyncMock(return_value=mock_asset_model),
    )

    response = client.post(
        "/api/v1/data/process/1",
        json={
            "file_id": "missing.txt",
            "chunk_size": 100,
            "overlap_size": 20,
            "do_reset": 0,
        },
    )

    assert response.status_code == 400
    assert response.json()["signal"] == ResponseSignal.FILE_ID_ERROR.value


def test_process_no_files(client, monkeypatch):
    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        mock_project_create_instance,
    )

    mock_asset_model = SimpleNamespace()
    mock_asset_model.get_all_project_assets = AsyncMock(return_value=[])

    monkeypatch.setattr(
        AssetModel,
        "create_instance",
        AsyncMock(return_value=mock_asset_model),
    )

    response = client.post(
        "/api/v1/data/process/1",
        json={
            "chunk_size": 100,
            "overlap_size": 20,
            "do_reset": 0,
        },
    )

    assert response.status_code == 400
    assert response.json()["signal"] == ResponseSignal.NO_FILES_ERROR.value


def test_process_success(client, monkeypatch):
    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        mock_project_create_instance,
    )

    mock_asset = make_asset(
        object_id=ASSET_ID,
        asset_name="test.txt",
    )

    mock_asset_model = SimpleNamespace()
    mock_asset_model.get_all_project_assets = AsyncMock(return_value=[mock_asset])

    monkeypatch.setattr(
        AssetModel,
        "create_instance",
        AsyncMock(return_value=mock_asset_model),
    )

    mock_chunk_model = SimpleNamespace()
    mock_chunk_model.insert_many_chunks = AsyncMock(return_value=2)
    mock_chunk_model.delete_chunks_by_project_id = AsyncMock(return_value=2)

    monkeypatch.setattr(
        ChunkModel,
        "create_instance",
        AsyncMock(return_value=mock_chunk_model),
    )

    def mock_get_file_content(self, file_id):
        return [
            SimpleNamespace(
                page_content="This is a test document.",
                metadata={"source": file_id},
            )
        ]

    def mock_process_file_content(self, file_content, file_id):
        return [
            SimpleNamespace(
                page_content="This is chunk one.",
                metadata={"file_id": file_id},
            ),
            SimpleNamespace(
                page_content="This is chunk two.",
                metadata={"file_id": file_id},
            ),
        ]

    monkeypatch.setattr(
        ProcessController,
        "get_file_content",
        mock_get_file_content,
    )

    monkeypatch.setattr(
        ProcessController,
        "process_file_content",
        mock_process_file_content,
    )

    response = client.post(
        "/api/v1/data/process/1",
        json={
            "chunk_size": 100,
            "overlap_size": 20,
            "do_reset": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["inserted_chunks"] == 2
    assert data["processed_files"] == 1


def test_process_success_with_reset(client, monkeypatch):
    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        mock_project_create_instance,
    )

    mock_asset = make_asset(
        object_id=ASSET_ID,
        asset_name="test.txt",
    )

    mock_asset_model = SimpleNamespace()
    mock_asset_model.get_all_project_assets = AsyncMock(return_value=[mock_asset])

    monkeypatch.setattr(
        AssetModel,
        "create_instance",
        AsyncMock(return_value=mock_asset_model),
    )

    mock_chunk_model = SimpleNamespace()
    mock_chunk_model.insert_many_chunks = AsyncMock(return_value=1)
    mock_chunk_model.delete_chunks_by_project_id = AsyncMock(return_value=5)

    monkeypatch.setattr(
        ChunkModel,
        "create_instance",
        AsyncMock(return_value=mock_chunk_model),
    )

    monkeypatch.setattr(
        ProcessController,
        "get_file_content",
        lambda self, file_id: [
            SimpleNamespace(
                page_content="Test content",
                metadata={"source": file_id},
            )
        ],
    )

    monkeypatch.setattr(
        ProcessController,
        "process_file_content",
        lambda self, file_content, file_id: [
            SimpleNamespace(
                page_content="Test chunk",
                metadata={"file_id": file_id},
            )
        ],
    )

    response = client.post(
        "/api/v1/data/process/1",
        json={
            "chunk_size": 100,
            "overlap_size": 20,
            "do_reset": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["inserted_chunks"] == 1
    assert response.json()["processed_files"] == 1

    mock_chunk_model.delete_chunks_by_project_id.assert_awaited_once_with(
        project_id=PROJECT_ID
    )


def test_upload_asset_creation_failure(client, monkeypatch):
    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        mock_project_create_instance,
    )

    async def mock_asset_create_instance(db_client):
        mock_model = SimpleNamespace()
        mock_model.create_asset = AsyncMock(side_effect=RuntimeError("database error"))
        return mock_model

    monkeypatch.setattr(
        AssetModel,
        "create_instance",
        mock_asset_create_instance,
    )

    monkeypatch.setattr(
        DataController,
        "validate_uploaded_file",
        lambda self, file: (True, "FILE_VALIDATED_SUCCESS"),
    )

    file_path = "/tmp/test_upload_failure.txt"

    monkeypatch.setattr(
        DataController,
        "generate_unique_filepath",
        lambda self, orig_file_name, project_id: (
            file_path,
            "test.txt",
        ),
    )

    response = client.post(
        "/api/v1/data/upload/1",
        files={
            "file": (
                "test.txt",
                b"Hello World",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["signal"] == ResponseSignal.FILE_UPLOAD_FAILED.value


def test_process_specific_file_success(client, monkeypatch):
    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        mock_project_create_instance,
    )

    mock_asset = make_asset(
        object_id=ASSET_ID,
        asset_name="test.txt",
    )

    mock_asset_model = SimpleNamespace()
    mock_asset_model.get_asset_record = AsyncMock(return_value=mock_asset)

    monkeypatch.setattr(
        AssetModel,
        "create_instance",
        AsyncMock(return_value=mock_asset_model),
    )

    mock_chunk_model = SimpleNamespace()
    mock_chunk_model.insert_many_chunks = AsyncMock(return_value=1)

    monkeypatch.setattr(
        ChunkModel,
        "create_instance",
        AsyncMock(return_value=mock_chunk_model),
    )

    monkeypatch.setattr(
        ProcessController,
        "get_file_content",
        lambda self, file_id: [
            SimpleNamespace(
                page_content="Test document",
                metadata={"source": file_id},
            )
        ],
    )

    monkeypatch.setattr(
        ProcessController,
        "process_file_content",
        lambda self, file_content, file_id: [
            SimpleNamespace(
                page_content="Test chunk",
                metadata={"file_id": file_id},
            )
        ],
    )

    response = client.post(
        "/api/v1/data/process/1",
        json={
            "file_id": "test.txt",
            "chunk_size": 100,
            "overlap_size": 20,
            "do_reset": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["inserted_chunks"] == 1
    assert data["processed_files"] == 1


def test_process_file_content_load_failure(client, monkeypatch):
    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        mock_project_create_instance,
    )

    mock_asset = make_asset(
        object_id=ASSET_ID,
        asset_name="test.txt",
    )

    mock_asset_model = SimpleNamespace()
    mock_asset_model.get_all_project_assets = AsyncMock(return_value=[mock_asset])

    monkeypatch.setattr(
        AssetModel,
        "create_instance",
        AsyncMock(return_value=mock_asset_model),
    )

    mock_chunk_model = SimpleNamespace()
    mock_chunk_model.insert_many_chunks = AsyncMock()

    monkeypatch.setattr(
        ChunkModel,
        "create_instance",
        AsyncMock(return_value=mock_chunk_model),
    )

    monkeypatch.setattr(
        ProcessController,
        "get_file_content",
        lambda self, file_id: None,
    )

    response = client.post(
        "/api/v1/data/process/1",
        json={
            "chunk_size": 100,
            "overlap_size": 20,
            "do_reset": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["inserted_chunks"] == 0
    assert data["processed_files"] == 0
    mock_chunk_model.insert_many_chunks.assert_not_awaited()


def test_process_no_chunks_generated(client, monkeypatch):
    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        mock_project_create_instance,
    )

    mock_asset = make_asset(
        object_id=ASSET_ID,
        asset_name="test.txt",
    )

    mock_asset_model = SimpleNamespace()
    mock_asset_model.get_all_project_assets = AsyncMock(return_value=[mock_asset])

    monkeypatch.setattr(
        AssetModel,
        "create_instance",
        AsyncMock(return_value=mock_asset_model),
    )

    mock_chunk_model = SimpleNamespace()
    mock_chunk_model.insert_many_chunks = AsyncMock()

    monkeypatch.setattr(
        ChunkModel,
        "create_instance",
        AsyncMock(return_value=mock_chunk_model),
    )

    monkeypatch.setattr(
        ProcessController,
        "get_file_content",
        lambda self, file_id: [
            SimpleNamespace(
                page_content="Test content",
                metadata={},
            )
        ],
    )

    monkeypatch.setattr(
        ProcessController,
        "process_file_content",
        lambda self, file_content, file_id: [],
    )

    response = client.post(
        "/api/v1/data/process/1",
        json={
            "chunk_size": 100,
            "overlap_size": 20,
            "do_reset": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["inserted_chunks"] == 0
    assert data["processed_files"] == 0
    mock_chunk_model.insert_many_chunks.assert_not_awaited()


def test_process_empty_chunk_is_skipped(client, monkeypatch):
    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        mock_project_create_instance,
    )

    mock_asset = make_asset(
        object_id=ASSET_ID,
        asset_name="test.txt",
    )

    mock_asset_model = SimpleNamespace()
    mock_asset_model.get_all_project_assets = AsyncMock(return_value=[mock_asset])

    monkeypatch.setattr(
        AssetModel,
        "create_instance",
        AsyncMock(return_value=mock_asset_model),
    )

    mock_chunk_model = SimpleNamespace()
    mock_chunk_model.insert_many_chunks = AsyncMock()

    monkeypatch.setattr(
        ChunkModel,
        "create_instance",
        AsyncMock(return_value=mock_chunk_model),
    )

    monkeypatch.setattr(
        ProcessController,
        "get_file_content",
        lambda self, file_id: [
            SimpleNamespace(
                page_content="Test content",
                metadata={},
            )
        ],
    )

    monkeypatch.setattr(
        ProcessController,
        "process_file_content",
        lambda self, file_content, file_id: [
            SimpleNamespace(
                page_content="   ",
                metadata={"file_id": file_id},
            )
        ],
    )

    response = client.post(
        "/api/v1/data/process/1",
        json={
            "chunk_size": 100,
            "overlap_size": 20,
            "do_reset": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["inserted_chunks"] == 0
    assert data["processed_files"] == 0
    mock_chunk_model.insert_many_chunks.assert_not_awaited()


def test_process_insert_chunks_failure(client, monkeypatch):
    monkeypatch.setattr(
        ProjectModel,
        "create_instance",
        mock_project_create_instance,
    )

    mock_asset = make_asset(
        object_id=ASSET_ID,
        asset_name="test.txt",
    )

    mock_asset_model = SimpleNamespace()
    mock_asset_model.get_all_project_assets = AsyncMock(return_value=[mock_asset])

    monkeypatch.setattr(
        AssetModel,
        "create_instance",
        AsyncMock(return_value=mock_asset_model),
    )

    mock_chunk_model = SimpleNamespace()
    mock_chunk_model.insert_many_chunks = AsyncMock(
        side_effect=RuntimeError("insert failed")
    )

    monkeypatch.setattr(
        ChunkModel,
        "create_instance",
        AsyncMock(return_value=mock_chunk_model),
    )

    monkeypatch.setattr(
        ProcessController,
        "get_file_content",
        lambda self, file_id: [
            SimpleNamespace(
                page_content="Test content",
                metadata={},
            )
        ],
    )

    monkeypatch.setattr(
        ProcessController,
        "process_file_content",
        lambda self, file_content, file_id: [
            SimpleNamespace(
                page_content="Test chunk",
                metadata={"file_id": file_id},
            )
        ],
    )

    response = client.post(
        "/api/v1/data/process/1",
        json={
            "chunk_size": 100,
            "overlap_size": 20,
            "do_reset": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["inserted_chunks"] == 0
    assert data["processed_files"] == 0

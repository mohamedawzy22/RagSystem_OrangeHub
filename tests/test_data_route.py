from fastapi.testclient import TestClient

from main import app
from controllers.DataController import DataController


client = TestClient(app)


def test_upload_invalid_file(monkeypatch):
    def mock_validate(self, file):
        return False, "FILE_TYPE_NOT_SUPPORTED"

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
    assert response.json()["signal"] == "FILE_TYPE_NOT_SUPPORTED"


def test_upload_success(tmp_path, monkeypatch):
    file_path = tmp_path / "test.txt"

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
    assert response.json()["file_name"] == "test.txt"
    assert file_path.read_bytes() == b"Hello World"


def test_upload_failed_to_save(tmp_path, monkeypatch):
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

    assert response.status_code == 500

import io
import os

from fastapi import UploadFile
from starlette.datastructures import Headers

from controllers.data_controller import DataController, ProjectController
from models import ResponseSignal


def test_valid_txt_file():
    file = UploadFile(
        filename="test.txt",
        file=io.BytesIO(b"Hello World"),
        headers=Headers({"content-type": "text/plain"}),
    )

    controller = DataController()

    is_valid, _ = controller.validate_uploaded_file(file)

    assert is_valid is True


def test_invalid_file_type():
    file = UploadFile(
        filename="test.exe",
        file=io.BytesIO(b"fake file"),
        headers=Headers({"content-type": "application/octet-stream"}),
    )

    controller = DataController()

    is_valid, _ = controller.validate_uploaded_file(file)

    assert is_valid is False


def test_cleaned_file_name():
    controller = DataController()

    result = controller.cleaned_file_name(" my file@#test.txt ")

    assert result == "myfiletest.txt"


def test_file_size_exceeded():
    file = UploadFile(
        filename="large.txt",
        file=io.BytesIO(b"large file"),
        headers=Headers({"content-type": "text/plain"}),
    )

    file.size = 11 * 1024 * 1024

    controller = DataController()

    is_valid, result = controller.validate_uploaded_file(file)

    assert is_valid is False
    assert result == ResponseSignal.FILE_SIZE_EXCEEDED.value


def test_get_project_path(tmp_path):
    controller = ProjectController()
    controller.files_dir = str(tmp_path)

    result = controller.get_project_path("123")

    assert result == os.path.join(str(tmp_path), "123")
    assert os.path.exists(result)


def test_validate_uploaded_file_missing():
    controller = DataController()

    result = controller.validate_uploaded_file(None)

    assert result == (
        False,
        ResponseSignal.FILE_NOT_FOUND.value,
    )


def test_generate_unique_filepath_collision(tmp_path, monkeypatch):
    controller = DataController()

    monkeypatch.setattr(
        ProjectController,
        "get_project_path",
        lambda self, project_id: str(tmp_path),
    )

    first_path = tmp_path / "abc_test.txt"
    first_path.write_text("existing")

    random_keys = iter(["abc", "def"])

    monkeypatch.setattr(
        controller,
        "generate_random_string",
        lambda: next(random_keys),
    )

    file_path, file_name = controller.generate_unique_filepath(
        orig_file_name="test.txt",
        project_id="project-1",
    )

    assert file_path == str(tmp_path / "def_test.txt")
    assert file_name == "def_test.txt"

import os
from types import SimpleNamespace

import pytest

from controllers.base_controller import BaseController
from controllers.process_controller import Document, ProcessController
from models.db_schemes import Asset, Project


def test_get_file_extension():
    controller = ProcessController("test_project")

    assert controller.get_file_extension("test.txt") == ".txt"
    assert controller.get_file_extension("test.PDF") == ".pdf"


def test_get_file_loader_file_not_found(monkeypatch):
    controller = ProcessController("test_project")

    monkeypatch.setattr(
        os.path,
        "exists",
        lambda path: False,
    )

    result = controller.get_file_loader("missing.txt")

    assert result is None


def test_get_file_loader_txt(tmp_path, monkeypatch):
    controller = ProcessController("test_project")

    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello world")

    monkeypatch.setattr(
        controller,
        "project_path",
        str(tmp_path),
    )

    result = controller.get_file_loader("test.txt")

    assert result is not None


def test_get_file_loader_pdf(tmp_path, monkeypatch):
    controller = ProcessController("test_project")

    file_path = tmp_path / "test.pdf"
    file_path.write_bytes(b"fake pdf")

    monkeypatch.setattr(
        controller,
        "project_path",
        str(tmp_path),
    )

    result = controller.get_file_loader("test.pdf")

    assert result == str(file_path)


def test_get_file_loader_unsupported(tmp_path, monkeypatch):
    controller = ProcessController("test_project")

    file_path = tmp_path / "test.docx"
    file_path.write_bytes(b"fake file")

    monkeypatch.setattr(
        controller,
        "project_path",
        str(tmp_path),
    )

    result = controller.get_file_loader("test.docx")

    assert result is None


def test_get_file_content_empty():
    controller = ProcessController("test_project")

    result = controller.process_file_content(
        file_content=[],
        file_id="test.txt",
    )

    assert result == []


def test_process_file_content():
    controller = ProcessController(
        project_id="test_project",
        chunk_size=20,
        overlap_size=5,
    )

    file_content = [
        Document(
            page_content="This is a simple test document with some words",
            metadata={"source": "test.txt"},
        )
    ]

    chunks = controller.process_file_content(
        file_content=file_content,
        file_id="test.txt",
    )

    assert chunks
    assert all(chunk.page_content.strip() for chunk in chunks)

    for index, chunk in enumerate(chunks):
        assert chunk.metadata["file_id"] == "test.txt"
        assert chunk.metadata["chunk_index"] == index
        assert "chunk_size" in chunk.metadata


def test_validate_chunk_config():
    ProcessController(
        project_id="test_project",
        chunk_size=100,
        overlap_size=20,
    )


def test_validate_chunk_config_invalid_size():
    with pytest.raises(ValueError, match="chunk_size must be greater than zero"):
        ProcessController(
            project_id="test_project",
            chunk_size=0,
            overlap_size=0,
        )


def test_validate_chunk_config_negative_overlap():
    with pytest.raises(ValueError, match="overlap_size cannot be negative"):
        ProcessController(
            project_id="test_project",
            chunk_size=100,
            overlap_size=-1,
        )


def test_validate_chunk_config_invalid_overlap():
    with pytest.raises(
        ValueError,
        match="overlap_size must be smaller than chunk_size",
    ):
        ProcessController(
            project_id="test_project",
            chunk_size=100,
            overlap_size=100,
        )


def test_process_file_content_none():
    controller = ProcessController("test_project")

    result = controller.process_file_content(
        file_content=None,
        file_id="test.txt",
    )

    assert result == []


def test_log_chunk_statistics_empty():
    ProcessController._log_chunk_statistics(
        chunks=[],
        file_id="test.txt",
    )


def test_get_file_content_missing_file(monkeypatch):
    controller = ProcessController("test_project")

    monkeypatch.setattr(
        controller,
        "get_file_loader",
        lambda file_id: None,
    )

    result = controller.get_file_content("missing.txt")

    assert result is None


def test_get_file_content_txt(monkeypatch):
    controller = ProcessController("test_project")

    document = SimpleNamespace(
        page_content="Hello world",
        metadata={"source": "test.txt"},
    )

    class MockLoader:
        def load(self):
            return [document]

    monkeypatch.setattr(
        controller,
        "get_file_loader",
        lambda file_id: MockLoader(),
    )

    result = controller.get_file_content("test.txt")

    assert len(result) == 1
    assert result[0].page_content == "Hello world"
    assert result[0].metadata == {"source": "test.txt"}


def test_get_file_content_txt_skips_empty(monkeypatch):
    controller = ProcessController("test_project")

    documents = [
        SimpleNamespace(
            page_content="   ",
            metadata={},
        ),
        SimpleNamespace(
            page_content="Hello",
            metadata={"source": "test.txt"},
        ),
    ]

    class MockLoader:
        def load(self):
            return documents

    monkeypatch.setattr(
        controller,
        "get_file_loader",
        lambda file_id: MockLoader(),
    )

    result = controller.get_file_content("test.txt")

    assert len(result) == 1
    assert result[0].page_content == "Hello"


def test_get_file_content_exception(monkeypatch):
    controller = ProcessController("test_project")

    class MockLoader:
        def load(self):
            raise RuntimeError("load failed")

    monkeypatch.setattr(
        controller,
        "get_file_loader",
        lambda file_id: MockLoader(),
    )

    result = controller.get_file_content("test.txt")

    assert result is None


def test_project_id_invalid():
    with pytest.raises(ValueError, match="project_id must be alphanumeric"):
        Project(project_id="test-project")


def test_generate_random_string():
    controller = BaseController()

    result = controller.generate_random_string(length=12)

    assert len(result) == 12
    assert result.islower()
    assert result.isalnum()


def test_project_id_validation_success():
    project = Project(project_id="project123")

    assert project.project_id == "project123"


def test_project_id_validation_invalid():
    with pytest.raises(
        ValueError,
        match="project_id must be alphanumeric",
    ):
        Project(project_id="project-123")


def test_project_get_indexes():
    indexes = Project.get_indexes()

    assert len(indexes) == 1
    assert indexes[0]["name"] == "project_id_index_1"
    assert indexes[0]["unique"] is True


def test_asset_get_indexes():
    indexes = Asset.get_indexes()

    assert len(indexes) == 2
    assert indexes[0]["name"] == "asset_project_id_index_1"
    assert indexes[1]["name"] == "asset_project_id_name_index_1"
    assert indexes[1]["unique"] is True

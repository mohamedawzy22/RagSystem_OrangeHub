import os

import pymupdf

from controllers.process_controller import ProcessController


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


def test_get_file_content_pdf(tmp_path):
    controller = ProcessController("test_project")

    file_path = tmp_path / "test.pdf"

    pdf = pymupdf.open()

    page = pdf.new_page()
    page.insert_text((72, 72), "First PDF page")

    page = pdf.new_page()
    page.insert_text((72, 72), "Second PDF page")

    pdf.save(str(file_path))
    pdf.close()

    controller.project_path = str(tmp_path)

    result = controller.get_file_content("test.pdf")

    assert result is not None
    assert len(result) == 2

    assert result[0].page_content == "First PDF page"
    assert result[1].page_content == "Second PDF page"

    assert result[0].metadata["source"] == "test.pdf"
    assert result[0].metadata["page"] == 1

    assert result[1].metadata["source"] == "test.pdf"
    assert result[1].metadata["page"] == 2


def test_get_file_loader_unsupported_extension_returns_none(tmp_path):
    controller = ProcessController("test_project")

    file_path = tmp_path / "test.docx"
    file_path.write_bytes(b"fake file")

    controller.project_path = str(tmp_path)

    result = controller.get_file_loader("test.docx")

    assert result is None

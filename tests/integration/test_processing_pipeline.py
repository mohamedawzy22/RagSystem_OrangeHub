from pathlib import Path

from controllers.process_controller import ProcessController


def test_processing_pipeline_txt(tmp_path):
    file_path = Path(tmp_path) / "document.txt"

    file_path.write_text(
        "Python is a programming language. "
        "FastAPI is a web framework for Python. "
        "RAG systems use retrieval and generation.",
        encoding="utf-8",
    )

    controller = ProcessController(
        project_id="integration-project",
        chunk_size=80,
        overlap_size=10,
    )

    controller.project_path = str(tmp_path)

    documents = controller.get_file_content("document.txt")

    assert documents
    assert len(documents) == 1
    assert documents[0].page_content.strip()

    chunks = controller.process_file_content(
        file_content=documents,
        file_id="document.txt",
    )

    assert chunks
    assert all(chunk.page_content.strip() for chunk in chunks)

    for index, chunk in enumerate(chunks):
        assert chunk.metadata["file_id"] == "document.txt"
        assert chunk.metadata["chunk_index"] == index
        assert "chunk_size" in chunk.metadata
        assert len(chunk.page_content) <= 80

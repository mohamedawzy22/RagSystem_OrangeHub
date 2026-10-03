import os
from dataclasses import dataclass
from statistics import mean, median

import pymupdf
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from models import ProcessingEnum
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    page_content: str
    metadata: dict


class DocumentProcessor:
    def __init__(
        self,
        project_path: str,
        chunk_size: int = 100,
        overlap_size: int = 20,
    ):
        self.project_path = project_path

        self._validate_chunk_config(
            chunk_size=chunk_size,
            overlap_size=overlap_size,
        )

        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

        logger.info(
            "Document processor initialized: chunk_size=%s | overlap=%s",
            self.chunk_size,
            self.overlap_size,
        )

    @staticmethod
    def _validate_chunk_config(
        chunk_size: int,
        overlap_size: int,
    ) -> None:
        if not isinstance(chunk_size, int):
            raise TypeError(
                "chunk_size must be an integer",
            )

        if not isinstance(overlap_size, int):
            raise TypeError(
                "overlap_size must be an integer",
            )

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero",
            )

        if overlap_size < 0:
            raise ValueError(
                "overlap_size cannot be negative",
            )

        if overlap_size >= chunk_size:
            raise ValueError(
                "overlap_size must be smaller than chunk_size",
            )

    @staticmethod
    def get_file_extension(
        file_id: str,
    ) -> str:
        return os.path.splitext(
            file_id,
        )[-1].lower()

    def get_file_loader(
        self,
        file_id: str,
    ):
        file_ext = self.get_file_extension(
            file_id=file_id,
        )

        file_path = os.path.join(
            self.project_path,
            file_id,
        )

        if not os.path.exists(file_path):
            logger.error(
                "File not found: file_id=%s | path=%s",
                file_id,
                file_path,
            )
            return None

        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(
                file_path,
                encoding="utf-8",
            )

        if file_ext == ProcessingEnum.PDF.value:
            return file_path

        logger.error(
            "Unsupported file extension: %s",
            file_ext,
        )

        return None

    def get_file_content(
        self,
        file_id: str,
    ) -> list[Document] | None:
        loader = self.get_file_loader(
            file_id=file_id,
        )

        if loader is None:
            return None

        try:
            file_ext = self.get_file_extension(
                file_id=file_id,
            )

            if file_ext == ProcessingEnum.TXT.value:
                documents = loader.load()

                return [
                    Document(
                        page_content=doc.page_content,
                        metadata=doc.metadata,
                    )
                    for doc in documents
                    if doc.page_content.strip()
                ]

            if file_ext == ProcessingEnum.PDF.value:
                pdf = pymupdf.open(loader)

                documents = []

                try:
                    for page_number, page in enumerate(pdf):
                        text = page.get_text(
                            "text",
                        ).strip()

                        if not text:
                            continue

                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": file_id,
                                    "page": page_number + 1,
                                },
                            )
                        )
                finally:
                    pdf.close()

                logger.info(
                    "Loaded PDF successfully: file_id=%s | pages=%s",
                    file_id,
                    len(documents),
                )

                return documents

        except Exception:
            logger.exception(
                "Failed to load file: file_id=%s",
                file_id,
            )

            return None

        return None

    def process_file_content(
        self,
        file_content: list[Document],
        file_id: str,
    ):
        logger.info(
            "Processing file: "
            "file_id=%s | chunk_size=%s characters | "
            "overlap=%s characters",
            file_id,
            self.chunk_size,
            self.overlap_size,
        )

        if not file_content:
            logger.warning(
                "No content found for file_id=%s",
                file_id,
            )

            return []

        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap_size,
            separator=" ",
            length_function=len,
        )

        file_content_texts = [record.page_content for record in file_content]

        file_content_metadata = [record.metadata for record in file_content]

        chunks = text_splitter.create_documents(
            file_content_texts,
            metadatas=file_content_metadata,
        )

        for index, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "file_id": file_id,
                    "chunk_index": index,
                    "chunk_size": len(
                        chunk.page_content,
                    ),
                }
            )

        self._log_chunk_statistics(
            chunks=chunks,
            file_id=file_id,
        )

        return chunks

    @staticmethod
    def _log_chunk_statistics(
        chunks: list,
        file_id: str,
    ) -> None:
        if not chunks:
            logger.warning(
                "No chunks generated for file_id=%s",
                file_id,
            )
            return

        lengths = [len(chunk.page_content) for chunk in chunks]

        logger.info(
            "Chunk statistics | "
            "file_id=%s | chunks=%s | "
            "min=%s characters | "
            "max=%s characters | "
            "mean=%.2f characters | "
            "median=%.2f characters",
            file_id,
            len(chunks),
            min(lengths),
            max(lengths),
            mean(lengths),
            median(lengths),
        )

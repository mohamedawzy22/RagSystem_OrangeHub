from enum import Enum


class RAGMessage(str, Enum):
    NO_CHUNKED_ASSETS = "No chunked assets found"
    NO_CHUNKS = "No chunks found"

    INDEX_SUCCESS = "Project indexed successfully"

    NO_RELEVANT_INFORMATION = (
        "I could not find relevant information to answer your question."
    )

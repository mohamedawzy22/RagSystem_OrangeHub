from enum import Enum


class ChatModelName(str, Enum):
    QWEN3 = "qwen3"
    OPENROUTER_MODEL = "openrouter_model"


class EmbeddingModelName(str, Enum):
    BGE_M3 = "bge-m3"
    OPENROUTER_EMBEDDING = "openrouter_embedding"

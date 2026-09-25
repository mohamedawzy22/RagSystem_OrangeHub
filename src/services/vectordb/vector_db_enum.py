from enum import Enum


class VectorDBProvider(str, Enum):
    QDRANT = "qdrant"


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    EUCLID = "euclid"
    DOT = "dot"

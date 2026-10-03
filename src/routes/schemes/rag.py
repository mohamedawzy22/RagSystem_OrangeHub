from pydantic import BaseModel, Field

QUERY_MAX_LENGTH = 4000
LIMIT_MAX_VALUE = 100


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=QUERY_MAX_LENGTH,
    )

    limit: int = Field(
        default=5,
        gt=0,
        le=LIMIT_MAX_VALUE,
    )


class SearchResult(BaseModel):
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class GenerateRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=QUERY_MAX_LENGTH,
    )

    limit: int = Field(
        default=5,
        gt=0,
        le=LIMIT_MAX_VALUE,
    )


class GenerateDocument(BaseModel):
    text: str
    score: float


class GenerateResponse(BaseModel):
    query: str
    answer: str
    documents: list[GenerateDocument]

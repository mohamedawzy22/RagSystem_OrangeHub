from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
    )

    limit: int = Field(
        default=5,
        gt=0,
        le=100,
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
    )

    limit: int = Field(
        default=5,
        gt=0,
        le=100,
    )


class GenerateDocument(BaseModel):
    text: str
    score: float


class GenerateResponse(BaseModel):
    query: str
    answer: str
    full_prompt: str
    documents: list[GenerateDocument]

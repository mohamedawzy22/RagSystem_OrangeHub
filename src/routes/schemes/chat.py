from pydantic import BaseModel, Field

from services.llm.llm_enum import ChatModelName


class ChatRequest(BaseModel):
    model: ChatModelName

    prompt: str = Field(min_length=1)

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
    )

    max_tokens: int | None = Field(
        default=None,
        gt=0,
    )


class ChatResponse(BaseModel):
    model: ChatModelName
    response: str

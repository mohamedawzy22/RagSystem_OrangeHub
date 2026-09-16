from abc import ABC, abstractmethod


class ChatModel(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        pass

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ):
        pass

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

    @abstractmethod
    async def health_check(self) -> bool:
        pass

    async def warm_up(self) -> None:
        await self.generate(
            prompt="Reply with OK",
            temperature=0.0,
            max_tokens=1,
        )

    async def close(self) -> None:
        return None

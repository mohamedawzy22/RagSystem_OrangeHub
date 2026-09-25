from abc import ABC, abstractmethod


class VectorDB(ABC):
    @abstractmethod
    async def create_collection(self) -> None:
        pass

    @abstractmethod
    async def delete_collection(self) -> None:
        pass

    @abstractmethod
    async def upsert(
        self,
        vectors: list[dict],
    ) -> None:
        pass

    @abstractmethod
    async def search(
        self,
        vector: list[float],
        limit: int = 5,
    ) -> list[dict]:
        pass

from bson import ObjectId

from .base_data_model import BaseDataModel
from .db_schemes import Asset
from .enums.database_enum import DataBaseEnum


class AssetModel(BaseDataModel):
    def __init__(
        self,
        db_client: object,
    ):
        super().__init__(
            db_client=db_client,
        )

        self.collection = self.db_client[DataBaseEnum.COLLECTION_ASSET_NAME.value]

    async def init_collection(self) -> None:
        """
        Initialize all required indexes for asset queries.

        Index creation is idempotent and safe to run
        during application startup.
        """

        indexes = Asset.get_indexes()

        for index in indexes:
            await self.collection.create_index(
                index["key"],
                name=index["name"],
                unique=index.get("unique", False),
            )

    async def create_asset(
        self,
        asset: Asset,
    ) -> Asset:
        result = await self.collection.insert_one(
            asset.model_dump(
                by_alias=True,
                exclude_unset=True,
            )
        )

        asset.id = result.inserted_id

        return asset

    async def get_all_project_assets(
        self,
        asset_project_id: ObjectId,
        asset_type: str,
    ) -> list[Asset]:
        records = await self.collection.find(
            {
                "asset_project_id": asset_project_id,
                "asset_type": asset_type,
            },
            projection={
                "_id": 1,
                "asset_project_id": 1,
                "asset_type": 1,
                "asset_name": 1,
                "asset_size": 1,
                "asset_config": 1,
                "asset_pushed_at": 1,
            },
        ).to_list(
            length=None,
        )

        return [Asset(**record) for record in records]

    async def get_asset_record(
        self,
        asset_project_id: ObjectId,
        asset_name: str,
    ) -> Asset | None:
        record = await self.collection.find_one(
            {
                "asset_project_id": asset_project_id,
                "asset_name": asset_name,
            },
            projection={
                "_id": 1,
                "asset_project_id": 1,
                "asset_type": 1,
                "asset_name": 1,
                "asset_size": 1,
                "asset_config": 1,
                "asset_pushed_at": 1,
            },
        )

        if record is None:
            return None

        return Asset(**record)

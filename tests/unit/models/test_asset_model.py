from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId

from models.asset_model import AssetModel
from models.enums.database_enum import DataBaseEnum


@pytest.fixture
def db_client():
    return MagicMock()


@pytest.fixture
def model(db_client):
    return AssetModel(db_client)


@pytest.mark.anyio
async def test_create_instance(db_client):
    with patch.object(
        AssetModel,
        "init_collection",
        new=AsyncMock(),
    ) as init_collection:
        instance = await AssetModel.create_instance(db_client)

    assert isinstance(instance, AssetModel)
    init_collection.assert_awaited_once()


@pytest.mark.anyio
async def test_init_collection_when_exists(model):
    model.db_client.list_collection_names = AsyncMock(
        return_value=[
            DataBaseEnum.COLLECTION_ASSET_NAME.value,
        ]
    )

    await model.init_collection()

    model.collection.create_index.assert_not_called()


@pytest.mark.anyio
async def test_init_collection_when_missing(model):
    model.db_client.list_collection_names = AsyncMock(
        return_value=[],
    )

    model.collection.create_index = AsyncMock()

    indexes = [
        {
            "key": "asset_project_id",
            "name": "asset_project_id_index_1",
            "unique": False,
        },
        {
            "key": [
                ("asset_project_id", 1),
                ("asset_name", 1),
            ],
            "name": "asset_project_id_name_index_1",
            "unique": True,
        },
    ]

    with patch(
        "models.asset_model.Asset.get_indexes",
        return_value=indexes,
    ):
        await model.init_collection()

    assert model.collection.create_index.await_count == 2


@pytest.mark.anyio
async def test_create_asset(model):
    asset = MagicMock()

    asset.model_dump.return_value = {
        "asset_name": "test.txt",
    }

    inserted_id = ObjectId()

    model.collection.insert_one = AsyncMock(
        return_value=MagicMock(
            inserted_id=inserted_id,
        )
    )

    result = await model.create_asset(asset)

    assert result is asset
    assert asset.id == inserted_id

    model.collection.insert_one.assert_awaited_once_with(
        {
            "asset_name": "test.txt",
        }
    )


@pytest.mark.anyio
async def test_get_all_project_assets_with_string_id(model):
    project_id = ObjectId()

    records = [
        {
            "_id": ObjectId(),
            "asset_project_id": project_id,
            "asset_name": "test.txt",
            "asset_type": "text/plain",
        }
    ]

    cursor = MagicMock()

    cursor.to_list = AsyncMock(
        return_value=records,
    )

    model.collection.find.return_value = cursor

    assets = [MagicMock()]

    with patch(
        "models.asset_model.Asset",
        side_effect=assets,
    ) as asset_class:
        result = await model.get_all_project_assets(
            str(project_id),
            "text/plain",
        )

    assert result == assets

    model.collection.find.assert_called_once_with(
        {
            "asset_project_id": project_id,
            "asset_type": "text/plain",
        }
    )

    asset_class.assert_called_once_with(**records[0])


@pytest.mark.anyio
async def test_get_all_project_assets_with_object_id(model):
    project_id = ObjectId()

    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=[])

    model.collection.find.return_value = cursor

    result = await model.get_all_project_assets(
        project_id,
        "text/plain",
    )

    assert result == []

    model.collection.find.assert_called_once_with(
        {
            "asset_project_id": project_id,
            "asset_type": "text/plain",
        }
    )


@pytest.mark.anyio
async def test_get_asset_record_found(model):
    project_id = ObjectId()

    record = {
        "_id": ObjectId(),
        "asset_project_id": project_id,
        "asset_name": "test.txt",
    }

    model.collection.find_one = AsyncMock(
        return_value=record,
    )

    asset = MagicMock()

    with patch(
        "models.asset_model.Asset",
        return_value=asset,
    ) as asset_class:
        result = await model.get_asset_record(
            str(project_id),
            "test.txt",
        )

    assert result is asset

    asset_class.assert_called_once_with(**record)


@pytest.mark.anyio
async def test_get_asset_record_not_found(model):
    project_id = ObjectId()

    model.collection.find_one = AsyncMock(
        return_value=None,
    )

    result = await model.get_asset_record(
        str(project_id),
        "missing.txt",
    )

    assert result is None

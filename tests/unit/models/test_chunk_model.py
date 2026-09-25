from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId

from models.chunk_model import ChunkModel
from models.enums.database_enum import DataBaseEnum


@pytest.fixture
def db_client():
    return MagicMock()


@pytest.fixture
def model(db_client):
    return ChunkModel(db_client)


@pytest.mark.anyio
async def test_create_instance(db_client):
    with patch.object(
        ChunkModel,
        "init_collection",
        new=AsyncMock(),
    ) as init_collection:
        instance = await ChunkModel.create_instance(db_client)

    assert isinstance(instance, ChunkModel)
    init_collection.assert_awaited_once()


@pytest.mark.anyio
async def test_init_collection_when_exists(model):
    model.db_client.list_collection_names = AsyncMock(
        return_value=[
            DataBaseEnum.COLLECTION_CHUNK_NAME.value,
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
            "key": "chunk_project_id",
            "name": "chunk_project_id_index_1",
            "unique": False,
        }
    ]

    with patch(
        "models.chunk_model.DataChunk.get_indexes",
        return_value=indexes,
    ):
        await model.init_collection()

    model.collection.create_index.assert_awaited_once_with(
        "chunk_project_id",
        name="chunk_project_id_index_1",
        unique=False,
    )


@pytest.mark.anyio
async def test_create_chunk(model):
    chunk = MagicMock()

    chunk.model_dump.return_value = {
        "chunk_text": "Hello world",
    }

    inserted_id = ObjectId()

    model.collection.insert_one = AsyncMock(
        return_value=MagicMock(
            inserted_id=inserted_id,
        )
    )

    result = await model.create_chunk(chunk)

    assert result is chunk
    assert chunk.id == inserted_id

    model.collection.insert_one.assert_awaited_once_with(
        {
            "chunk_text": "Hello world",
        }
    )


@pytest.mark.anyio
async def test_get_chunk_found(model):
    chunk_id = ObjectId()

    record = {
        "_id": chunk_id,
        "chunk_text": "Hello world",
    }

    model.collection.find_one = AsyncMock(
        return_value=record,
    )

    chunk = MagicMock()

    with patch(
        "models.chunk_model.DataChunk",
        return_value=chunk,
    ) as data_chunk_class:
        result = await model.get_chunk(
            str(chunk_id),
        )

    assert result is chunk

    model.collection.find_one.assert_awaited_once_with(
        {
            "_id": chunk_id,
        }
    )

    data_chunk_class.assert_called_once_with(**record)


@pytest.mark.anyio
async def test_get_chunk_not_found(model):
    chunk_id = ObjectId()

    model.collection.find_one = AsyncMock(
        return_value=None,
    )

    result = await model.get_chunk(
        str(chunk_id),
    )

    assert result is None


@pytest.mark.anyio
async def test_insert_many_chunks(model):
    chunk_one = MagicMock()
    chunk_two = MagicMock()
    chunk_three = MagicMock()

    chunk_one.model_dump.return_value = {"chunk_text": "one"}
    chunk_two.model_dump.return_value = {"chunk_text": "two"}
    chunk_three.model_dump.return_value = {"chunk_text": "three"}

    model.collection.bulk_write = AsyncMock()

    result = await model.insert_many_chunks(
        [
            chunk_one,
            chunk_two,
            chunk_three,
        ],
        batch_size=2,
    )

    assert result == 3
    assert model.collection.bulk_write.await_count == 2

    first_operations = model.collection.bulk_write.await_args_list[0].args[0]
    second_operations = model.collection.bulk_write.await_args_list[1].args[0]

    assert len(first_operations) == 2
    assert len(second_operations) == 1


@pytest.mark.anyio
async def test_insert_many_chunks_empty(model):
    model.collection.bulk_write = AsyncMock()

    result = await model.insert_many_chunks([])

    assert result == 0
    model.collection.bulk_write.assert_not_called()


@pytest.mark.anyio
async def test_delete_chunks_by_project_id(model):
    project_id = ObjectId()

    model.collection.delete_many = AsyncMock(
        return_value=MagicMock(
            deleted_count=4,
        )
    )

    result = await model.delete_chunks_by_project_id(
        project_id,
    )

    assert result == 4

    model.collection.delete_many.assert_awaited_once_with(
        {
            "chunk_project_id": project_id,
        }
    )


@pytest.mark.anyio
async def test_get_project_chunks(model):
    project_id = ObjectId()

    records = [
        {
            "_id": ObjectId(),
            "chunk_project_id": project_id,
            "chunk_text": "one",
        },
        {
            "_id": ObjectId(),
            "chunk_project_id": project_id,
            "chunk_text": "two",
        },
    ]

    cursor = MagicMock()

    cursor.skip.return_value = cursor
    cursor.limit.return_value = cursor
    cursor.to_list = AsyncMock(
        return_value=records,
    )

    model.collection.find.return_value = cursor

    chunks = [
        MagicMock(),
        MagicMock(),
    ]

    with patch(
        "models.chunk_model.DataChunk",
        side_effect=chunks,
    ) as data_chunk_class:
        result = await model.get_project_chunks(
            project_id,
            page_no=2,
            page_size=10,
        )

    assert result == chunks

    model.collection.find.assert_called_once_with(
        {
            "chunk_project_id": project_id,
        }
    )

    cursor.skip.assert_called_once_with(10)
    cursor.limit.assert_called_once_with(10)

    assert data_chunk_class.call_count == 2

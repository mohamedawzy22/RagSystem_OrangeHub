from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.enums.database_enum import DataBaseEnum
from models.project_model import ProjectModel


@pytest.fixture
def db_client():
    return MagicMock()


@pytest.fixture
def model(db_client):
    return ProjectModel(db_client)


@pytest.mark.anyio
async def test_create_instance(db_client):
    with patch.object(
        ProjectModel,
        "init_collection",
        new=AsyncMock(),
    ) as init_collection:
        instance = await ProjectModel.create_instance(db_client)

    assert isinstance(instance, ProjectModel)
    init_collection.assert_awaited_once()


@pytest.mark.anyio
async def test_init_collection_when_exists(model):
    model.db_client.list_collection_names = AsyncMock(
        return_value=[
            DataBaseEnum.COLLECTION_PROJECT_NAME.value,
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
            "key": "project_id",
            "name": "project_id_index_1",
            "unique": True,
        }
    ]

    with patch(
        "models.project_model.Project.get_indexes",
        return_value=indexes,
    ):
        await model.init_collection()

    model.collection.create_index.assert_awaited_once_with(
        "project_id",
        name="project_id_index_1",
        unique=True,
    )


@pytest.mark.anyio
async def test_create_project(model):
    project = MagicMock()

    project.model_dump.return_value = {
        "project_id": "project-1",
    }

    inserted_id = "generated-id"

    model.collection.insert_one = AsyncMock(
        return_value=MagicMock(
            inserted_id=inserted_id,
        )
    )

    result = await model.create_project(project)

    assert result is project
    assert project.id == inserted_id

    project.model_dump.assert_called_once_with(
        by_alias=True,
        exclude_unset=True,
    )

    model.collection.insert_one.assert_awaited_once_with(
        {
            "project_id": "project-1",
        }
    )


@pytest.mark.anyio
async def test_get_project_or_create_one_existing(model):
    record = {
        "_id": "project-id",
        "project_id": "project-1",
    }

    model.collection.find_one = AsyncMock(
        return_value=record,
    )

    project = MagicMock()

    with patch(
        "models.project_model.Project",
        return_value=project,
    ) as project_class:
        result = await model.get_project_or_create_one(
            "project-1",
        )

    assert result is project

    model.collection.find_one.assert_awaited_once_with(
        {
            "project_id": "project-1",
        }
    )

    project_class.assert_called_once_with(**record)


@pytest.mark.anyio
async def test_get_project_or_create_one_creates_when_missing(model):
    model.collection.find_one = AsyncMock(
        return_value=None,
    )

    created_project = MagicMock()

    model.create_project = AsyncMock(
        return_value=created_project,
    )

    project_instance = MagicMock()

    with patch(
        "models.project_model.Project",
        return_value=project_instance,
    ) as project_class:
        result = await model.get_project_or_create_one(
            "project-1",
        )

    assert result is created_project

    project_class.assert_called_once_with(
        project_id="project-1",
    )

    model.create_project.assert_awaited_once_with(
        project=project_instance,
    )


@pytest.mark.anyio
async def test_get_all_projects(model):
    records = [
        {
            "_id": "project-1",
            "project_id": "project-1",
        },
        {
            "_id": "project-2",
            "project_id": "project-2",
        },
    ]

    model.collection.count_documents = AsyncMock(
        return_value=12,
    )

    class AsyncCursor:
        def __init__(self, documents):
            self.documents = documents

        def skip(self, value):
            self.skip_value = value
            return self

        def limit(self, value):
            self.limit_value = value
            return self

        def __aiter__(self):
            return self._iterator()

        async def _iterator(self):
            for document in self.documents:
                yield document

    cursor = AsyncCursor(records)

    model.collection.find.return_value = cursor

    project_instances = [
        MagicMock(),
        MagicMock(),
    ]

    with patch(
        "models.project_model.Project",
        side_effect=project_instances,
    ) as project_class:
        projects, total_pages = await model.get_all_projects(
            page=2,
            page_size=5,
        )

    assert projects == project_instances
    assert total_pages == 3

    model.collection.count_documents.assert_awaited_once_with({})

    model.collection.find.assert_called_once_with()

    assert cursor.skip_value == 5
    assert cursor.limit_value == 5

    assert project_class.call_count == 2

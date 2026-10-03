from pymongo import ASCENDING, ReturnDocument

from .base_data_model import BaseDataModel
from .db_schemes import Project
from .enums.database_enum import DataBaseEnum


class ProjectModel(BaseDataModel):
    def __init__(
        self,
        db_client: object,
    ):
        super().__init__(
            db_client=db_client,
        )

        self.collection = self.db_client[DataBaseEnum.COLLECTION_PROJECT_NAME.value]

    async def init_collection(self) -> None:
        """
        Initialize indexes required by project queries.

        Index creation is idempotent and safe to run
        during application startup.
        """

        indexes = Project.get_indexes()

        for index in indexes:
            await self.collection.create_index(
                index["key"],
                name=index["name"],
                unique=index.get("unique", False),
            )

    async def create_project(
        self,
        project: Project,
    ) -> Project:
        result = await self.collection.insert_one(
            project.model_dump(
                by_alias=True,
                exclude_unset=True,
            )
        )

        project.id = result.inserted_id

        return project

    async def get_project_or_create_one(
        self,
        project_id: str,
    ) -> Project:
        """
        Resolve a project ID atomically.

        Uses upsert instead of find-then-insert so concurrent
        requests cannot race during project creation.
        """

        document = await self.collection.find_one_and_update(
            {"project_id": project_id},
            {
                "$setOnInsert": {
                    "project_id": project_id,
                },
            },
            projection={
                "_id": 1,
                "project_id": 1,
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

        if document is None:
            raise RuntimeError(
                f"Failed to resolve project: {project_id}",
            )

        return Project(**document)

    async def get_all_projects(
        self,
        page: int = 1,
        page_size: int = 10,
    ):
        total_documents = await self.collection.count_documents({})

        total_pages = total_documents // page_size

        if total_documents % page_size > 0:
            total_pages += 1

        cursor = (
            self.collection.find(
                {},
                projection={
                    "_id": 1,
                    "project_id": 1,
                },
            )
            .sort(
                "_id",
                ASCENDING,
            )
            .skip(
                (page - 1) * page_size,
            )
            .limit(
                page_size,
            )
        )

        projects = []

        async for document in cursor:
            projects.append(Project(**document))

        return projects, total_pages

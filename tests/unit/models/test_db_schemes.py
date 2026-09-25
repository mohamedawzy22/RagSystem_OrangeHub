import pytest

from models.db_schemes import Asset, Project


def test_project_id_validation_success():
    project = Project(project_id="project123")

    assert project.project_id == "project123"


def test_project_id_validation_invalid():
    with pytest.raises(
        ValueError,
        match="project_id must be alphanumeric",
    ):
        Project(project_id="project-123")


def test_project_get_indexes():
    indexes = Project.get_indexes()

    assert len(indexes) == 1
    assert indexes[0]["name"] == "project_id_index_1"
    assert indexes[0]["unique"] is True


def test_asset_get_indexes():
    indexes = Asset.get_indexes()

    assert len(indexes) == 2
    assert indexes[0]["name"] == "asset_project_id_index_1"
    assert indexes[1]["name"] == "asset_project_id_name_index_1"
    assert indexes[1]["unique"] is True

from types import SimpleNamespace


def make_project(
    object_id="project-db-id",
    project_id="project-1",
):
    return SimpleNamespace(
        id=object_id,
        project_id=project_id,
    )

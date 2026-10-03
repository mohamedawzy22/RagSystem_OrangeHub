from main import app


def test_api_contains_required_endpoints():
    paths = app.openapi()["paths"]

    assert "/api/v1/health" in paths

    assert "/api/v1/data/upload/{project_id}" in paths
    assert "/api/v1/data/process/{project_id}" in paths

    assert "/api/v1/rag/index/{project_id}" in paths
    assert "/api/v1/rag/search/{project_id}" in paths
    assert "/api/v1/rag/generate/{project_id}" in paths


def test_rag_endpoints_are_post():
    paths = app.openapi()["paths"]

    assert "post" in paths["/api/v1/rag/index/{project_id}"]
    assert "post" in paths["/api/v1/rag/search/{project_id}"]
    assert "post" in paths["/api/v1/rag/generate/{project_id}"]

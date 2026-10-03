from types import SimpleNamespace

import pytest


@pytest.fixture
def sample_documents() -> list[dict]:
    return [
        {
            "text": "Lebanon is a country in the Middle East.",
            "score": 0.91,
        },
        {
            "text": "Beirut is the capital of Lebanon.",
            "score": 0.84,
        },
    ]


@pytest.fixture
def sample_chunks() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            id="chunk-1",
            chunk_text="Lebanon is a country in the Middle East.",
            chunk_asset_id="asset-1",
        ),
        SimpleNamespace(
            id="chunk-2",
            chunk_text="Beirut is the capital of Lebanon.",
            chunk_asset_id="asset-1",
        ),
    ]

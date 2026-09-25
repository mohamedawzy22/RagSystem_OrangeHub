from unittest.mock import MagicMock

import pytest

from services.vectordb.vector_db_enum import VectorDBProvider
from services.vectordb.vectordb_manager import VectorDBManager


def test_load_databases():
    database = MagicMock()

    factory = MagicMock()
    factory.create.return_value = database

    manager = VectorDBManager(factory)

    manager.load_databases()

    assert manager._databases["default"] is database

    factory.create.assert_called_once_with(
        VectorDBProvider.QDRANT,
    )


def test_get_database_success():
    database = MagicMock()

    factory = MagicMock()

    manager = VectorDBManager(factory)

    manager._databases["default"] = database

    result = manager.get_database("default")

    assert result is database


def test_get_database_not_configured():
    factory = MagicMock()

    manager = VectorDBManager(factory)

    with pytest.raises(
        ValueError,
        match="Vector database 'default' is not configured",
    ):
        manager.get_database("default")

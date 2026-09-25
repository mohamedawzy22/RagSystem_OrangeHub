from unittest.mock import MagicMock

from models.base_data_model import BaseDataModel


def test_base_data_model_initialization():
    db_client = MagicMock()

    model = BaseDataModel(db_client=db_client)

    assert model.db_client is db_client
    assert model.app_settings is not None

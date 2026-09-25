from models.db_schemes.rag_system.data_chunk import DataChunk


def test_data_chunk_indexes():
    indexes = DataChunk.get_indexes()

    assert indexes == [
        {
            "key": [("chunk_project_id", 1)],
            "name": "chunk_project_id_index_1",
            "unique": False,
        }
    ]

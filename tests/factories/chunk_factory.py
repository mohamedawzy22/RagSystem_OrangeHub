from types import SimpleNamespace


def make_chunk(
    object_id="chunk-1",
    asset_id="asset-1",
    text="Test chunk",
):
    return SimpleNamespace(
        id=object_id,
        chunk_asset_id=asset_id,
        chunk_text=text,
    )

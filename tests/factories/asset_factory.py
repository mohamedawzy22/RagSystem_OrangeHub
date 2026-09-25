from types import SimpleNamespace


def make_asset(
    object_id="asset-1",
    asset_name="test.txt",
):
    return SimpleNamespace(
        id=object_id,
        asset_name=asset_name,
    )

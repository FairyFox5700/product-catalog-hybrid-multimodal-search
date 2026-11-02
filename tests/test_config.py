from catalog_search.config import settings


def test_settings_loaded_from_env():
    assert settings.search_endpoint.startswith("https://")
    assert settings.search_index_name == "product-catalog-test"
    assert settings.embedding_dimensions == 8


def test_storage_and_vision_settings_present():
    assert settings.vision_endpoint.startswith("https://")
    assert settings.storage_container == "product-images-test"

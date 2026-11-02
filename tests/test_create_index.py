from catalog_search.config import settings
from catalog_search.create_index import (
    SEMANTIC_CONFIG_NAME,
    VECTOR_SEARCH_PROFILE,
    build_index_definition,
)


def test_index_has_expected_field_names():
    index = build_index_definition()
    field_names = {f.name for f in index.fields}
    assert field_names == {
        "id",
        "name",
        "description",
        "category",
        "brand",
        "price",
        "in_stock",
        "image_url",
        "image_blob_name",
        "text_vector",
        "image_vector",
    }


def test_vector_fields_use_configured_dimensions_and_profile():
    index = build_index_definition()
    vector_fields = {f.name: f for f in index.fields if f.name in ("text_vector", "image_vector")}

    for name, field in vector_fields.items():
        assert field.vector_search_dimensions == settings.embedding_dimensions
        assert field.vector_search_profile_name == VECTOR_SEARCH_PROFILE


def test_semantic_configuration_prioritizes_name_and_description():
    index = build_index_definition()
    config = index.semantic_search.configurations[0]

    assert config.name == SEMANTIC_CONFIG_NAME
    assert config.prioritized_fields.title_field.field_name == "name"
    content_fields = [f.field_name for f in config.prioritized_fields.content_fields]
    assert "description" in content_fields


def test_index_name_comes_from_settings():
    index = build_index_definition()
    assert index.name == settings.search_index_name

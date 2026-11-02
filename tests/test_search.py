"""Tests for the hybrid + multimodal query functions. The Azure AI Search client
and the embedding client are both mocked, so results are deterministic and no
network/live resources are required."""
from unittest.mock import MagicMock, patch

from catalog_search import search as search_module


def _fake_hit(id_, name):
    return {
        "id": id_,
        "name": name,
        "description": "desc",
        "category": "Shoes",
        "brand": "Northline",
        "price": 99.0,
        "in_stock": True,
        "image_url": f"https://dummy/{id_}.jpg",
    }


@patch("catalog_search.search._get_search_client")
@patch("catalog_search.search.embedding_client")
def test_search_by_text_passes_search_text_and_vector_query(mock_embed, mock_get_client):
    mock_embed.embed_text.return_value = [0.1] * 8
    mock_client = MagicMock()
    mock_client.search.return_value = [_fake_hit("1", "Trail Runner")]
    mock_get_client.return_value = mock_client

    results = search_module.search_by_text("red running shoes", top=5)

    assert results == [_fake_hit("1", "Trail Runner")]
    mock_embed.embed_text.assert_called_once_with("red running shoes")
    _, kwargs = mock_client.search.call_args
    assert kwargs["search_text"] == "red running shoes"
    assert kwargs["top"] == 5
    assert kwargs["query_type"] == "semantic"
    vector_query = kwargs["vector_queries"][0]
    assert vector_query.fields == "image_vector,text_vector"


@patch("catalog_search.search._get_search_client")
@patch("catalog_search.search.embedding_client")
def test_search_by_text_can_disable_semantic_ranking(mock_embed, mock_get_client):
    mock_embed.embed_text.return_value = [0.1] * 8
    mock_client = MagicMock()
    mock_client.search.return_value = []
    mock_get_client.return_value = mock_client

    search_module.search_by_text("shoes", use_semantic_ranking=False)

    _, kwargs = mock_client.search.call_args
    assert "query_type" not in kwargs


@patch("catalog_search.search._get_search_client")
@patch("catalog_search.search.embedding_client")
def test_search_by_image_uses_pure_vector_query_without_semantic_ranking(mock_embed, mock_get_client):
    mock_embed.embed_image_bytes.return_value = [0.2] * 8
    mock_client = MagicMock()
    mock_client.search.return_value = [_fake_hit("2", "Leather Loafer")]
    mock_get_client.return_value = mock_client

    results = search_module.search_by_image(b"fake-bytes", top=3)

    assert results == [_fake_hit("2", "Leather Loafer")]
    _, kwargs = mock_client.search.call_args
    assert kwargs["search_text"] == "*"
    assert "query_type" not in kwargs
    vector_query = kwargs["vector_queries"][0]
    assert vector_query.fields == "image_vector"


@patch("catalog_search.search._get_search_client")
@patch("catalog_search.search.embedding_client")
def test_search_combined_with_text_and_image_builds_two_vector_queries(mock_embed, mock_get_client):
    mock_embed.embed_text.return_value = [0.1] * 8
    mock_embed.embed_image_bytes.return_value = [0.2] * 8
    mock_client = MagicMock()
    mock_client.search.return_value = []
    mock_get_client.return_value = mock_client

    search_module.search_combined(text="but in blue", image_bytes=b"photo-bytes", top=10)

    _, kwargs = mock_client.search.call_args
    assert len(kwargs["vector_queries"]) == 2
    assert kwargs["query_type"] == "semantic"  # text was provided


@patch("catalog_search.search._get_search_client")
@patch("catalog_search.search.embedding_client")
def test_search_combined_image_only_skips_semantic_ranking(mock_embed, mock_get_client):
    mock_embed.embed_image_bytes.return_value = [0.2] * 8
    mock_client = MagicMock()
    mock_client.search.return_value = []
    mock_get_client.return_value = mock_client

    search_module.search_combined(image_bytes=b"photo-bytes")

    _, kwargs = mock_client.search.call_args
    assert len(kwargs["vector_queries"]) == 1
    assert "query_type" not in kwargs

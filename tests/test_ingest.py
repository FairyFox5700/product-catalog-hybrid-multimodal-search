"""Tests for the ingest pipeline — image store, embedding client, and search
client are all mocked so no real Azure resources are touched."""
from unittest.mock import MagicMock, patch

from catalog_search.ingest import CatalogItem, ingest_items, load_catalog


def test_load_catalog_parses_json(tmp_path):
    catalog_json = tmp_path / "catalog.json"
    catalog_json.write_text(
        """
        [
          {"name": "Shoe", "description": "desc", "category": "Shoes",
           "brand": "Northline", "price": 10.0, "image_path": "x.jpg"}
        ]
        """
    )

    items = load_catalog(catalog_json)

    assert len(items) == 1
    assert isinstance(items[0], CatalogItem)
    assert items[0].name == "Shoe"
    assert items[0].in_stock is True  # default


@patch("catalog_search.ingest._get_search_client")
@patch("catalog_search.ingest.embedding_client")
@patch("catalog_search.ingest.image_store")
def test_ingest_items_uploads_embeds_and_indexes(mock_image_store, mock_embed, mock_get_client):
    mock_image_store.upload_image.return_value = "abc.jpg"
    mock_image_store.get_url.return_value = "https://dummy/abc.jpg"
    mock_embed.embed_image_url.return_value = [0.1] * 8
    mock_embed.embed_text.return_value = [0.2] * 8

    mock_client = MagicMock()
    mock_client.upload_documents.return_value = [MagicMock(succeeded=True)]
    mock_get_client.return_value = mock_client

    item = CatalogItem(
        id="fixed-id",
        name="Trail Runner",
        description="A sneaker",
        category="Shoes",
        brand="Northline",
        price=89.99,
        image_path="trail_runner.jpg",
    )

    ingest_items([item])

    mock_image_store.upload_image.assert_called_once_with("trail_runner.jpg", blob_name="fixed-id.jpg")
    mock_embed.embed_image_url.assert_called_once_with("https://dummy/abc.jpg")
    mock_embed.embed_text.assert_called_once()
    mock_client.upload_documents.assert_called_once()
    (uploaded_docs,) = mock_client.upload_documents.call_args.kwargs.values()
    assert uploaded_docs[0]["id"] == "fixed-id"
    assert uploaded_docs[0]["image_vector"] == [0.1] * 8
    assert uploaded_docs[0]["text_vector"] == [0.2] * 8


@patch("catalog_search.ingest._get_search_client")
@patch("catalog_search.ingest.embedding_client")
@patch("catalog_search.ingest.image_store")
def test_ingest_items_warns_on_failed_documents(mock_image_store, mock_embed, mock_get_client, capsys):
    mock_image_store.upload_image.return_value = "abc.jpg"
    mock_image_store.get_url.return_value = "https://dummy/abc.jpg"
    mock_embed.embed_image_url.return_value = [0.0] * 8
    mock_embed.embed_text.return_value = [0.0] * 8

    mock_client = MagicMock()
    mock_client.upload_documents.return_value = [MagicMock(succeeded=False)]
    mock_get_client.return_value = mock_client

    item = CatalogItem(
        id="fail-id", name="Bad Item", description="x", category="Shoes",
        brand="Northline", price=1.0, image_path="x.jpg",
    )

    ingest_items([item])

    captured = capsys.readouterr()
    assert "WARNING" in captured.out

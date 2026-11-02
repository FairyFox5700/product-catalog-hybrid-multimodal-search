"""Tests for ImageStore — Blob Storage SDK calls are mocked, so no real
storage account is needed, and construction never touches the network."""
from unittest.mock import MagicMock, patch

from catalog_search.image_store import ImageStore


@patch("catalog_search.image_store.BlobServiceClient")
def test_construction_does_not_touch_network(mock_blob_service_client_cls):
    """Regression test: ImageStore() must not call the network in __init__."""
    ImageStore()
    mock_blob_service_client_cls.from_connection_string.assert_not_called()


@patch("catalog_search.image_store.BlobServiceClient")
def test_upload_image_creates_container_lazily_and_uploads(mock_blob_service_client_cls, tmp_path):
    mock_service_client = MagicMock()
    mock_blob_service_client_cls.from_connection_string.return_value = mock_service_client
    mock_container = MagicMock()
    mock_container.exists.return_value = False
    mock_service_client.get_container_client.return_value = mock_container

    image_file = tmp_path / "shoe.jpg"
    image_file.write_bytes(b"fake-image-bytes")

    store = ImageStore()
    blob_name = store.upload_image(image_file, blob_name="products/shoe.jpg")

    assert blob_name == "products/shoe.jpg"
    mock_container.create_container.assert_called_once()
    mock_container.upload_blob.assert_called_once()
    _, kwargs = mock_container.upload_blob.call_args
    assert kwargs["name"] == "products/shoe.jpg"
    assert kwargs["overwrite"] is True


@patch("catalog_search.image_store.BlobServiceClient")
def test_upload_image_skips_create_container_when_it_already_exists(mock_blob_service_client_cls, tmp_path):
    mock_service_client = MagicMock()
    mock_blob_service_client_cls.from_connection_string.return_value = mock_service_client
    mock_container = MagicMock()
    mock_container.exists.return_value = True
    mock_service_client.get_container_client.return_value = mock_container

    image_file = tmp_path / "shoe.jpg"
    image_file.write_bytes(b"fake-image-bytes")

    store = ImageStore()
    store.upload_image(image_file)

    mock_container.create_container.assert_not_called()


@patch("catalog_search.image_store.generate_blob_sas", return_value="sv=sas-token")
@patch("catalog_search.image_store.BlobServiceClient")
def test_get_url_builds_sas_url_with_account_key(mock_blob_service_client_cls, mock_generate_sas, tmp_path):
    mock_service_client = MagicMock()
    mock_service_client.account_name = "dummystorage"
    mock_service_client.credential.account_key = "ZHVtbXlrZXk="
    mock_blob_service_client_cls.from_connection_string.return_value = mock_service_client

    mock_container = MagicMock()
    mock_container.exists.return_value = True
    mock_blob_client = MagicMock()
    mock_blob_client.url = "https://dummystorage.blob.core.windows.net/product-images-test/shoe.jpg"
    mock_container.get_blob_client.return_value = mock_blob_client
    mock_service_client.get_container_client.return_value = mock_container

    store = ImageStore()
    url = store.get_url("shoe.jpg")

    assert url == f"{mock_blob_client.url}?sv=sas-token"
    mock_generate_sas.assert_called_once()

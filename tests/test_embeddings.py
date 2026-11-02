"""Tests for the multimodal embedding client — all HTTP calls are mocked, so no
real Azure AI Vision resource is needed."""
from unittest.mock import MagicMock, patch

from catalog_search.embeddings import MultimodalEmbeddingClient


def _mock_response(vector):
    resp = MagicMock()
    resp.json.return_value = {"vector": vector}
    resp.raise_for_status.return_value = None
    return resp


@patch("catalog_search.embeddings.requests.post")
def test_embed_text_calls_vectorize_text_endpoint(mock_post):
    mock_post.return_value = _mock_response([0.1] * 8)
    client = MultimodalEmbeddingClient()

    vector = client.embed_text("red running shoes")

    assert vector == [0.1] * 8
    called_url = mock_post.call_args.args[0]
    assert called_url.endswith("retrieval:vectorizeText")
    assert mock_post.call_args.kwargs["json"] == {"text": "red running shoes"}


@patch("catalog_search.embeddings.requests.post")
def test_embed_image_bytes_calls_vectorize_image_endpoint(mock_post):
    mock_post.return_value = _mock_response([0.2] * 8)
    client = MultimodalEmbeddingClient()

    vector = client.embed_image_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")

    assert vector == [0.2] * 8
    called_url = mock_post.call_args.args[0]
    assert called_url.endswith("retrieval:vectorizeImage")
    assert mock_post.call_args.kwargs["data"] == b"\xff\xd8\xff\xe0fake-jpeg-bytes"


@patch("catalog_search.embeddings.requests.post")
def test_embed_image_url_sends_json_body(mock_post):
    mock_post.return_value = _mock_response([0.3] * 8)
    client = MultimodalEmbeddingClient()

    vector = client.embed_image_url("https://example.com/shoe.jpg")

    assert vector == [0.3] * 8
    assert mock_post.call_args.kwargs["json"] == {"url": "https://example.com/shoe.jpg"}


@patch("catalog_search.embeddings.requests.post")
def test_uses_subscription_key_header_when_api_key_present(mock_post):
    mock_post.return_value = _mock_response([0.0] * 8)
    client = MultimodalEmbeddingClient()

    client.embed_text("hello")

    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Ocp-Apim-Subscription-Key"] == "dummy-vision-key"

"""
Multimodal embeddings via Azure AI Vision's Image Retrieval API.

This is the "Azure Cognitive Services" multimodal embedding model (Florence-based):
it projects BOTH images and free-text queries into the *same* vector space, which is
what enables true cross-modal search (e.g. a text query retrieving the most visually
similar product photos, or an uploaded photo retrieving textually-described products).

Docs: https://learn.microsoft.com/azure/ai-services/computer-vision/how-to/image-retrieval
"""
from __future__ import annotations

from typing import List

import requests
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

from .config import settings

_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


class MultimodalEmbeddingClient:
    """Thin wrapper around the Vision 4.0 vectorizeImage / vectorizeText endpoints."""

    def __init__(self) -> None:
        self._endpoint = settings.vision_endpoint.rstrip("/")
        self._api_version = settings.vision_api_version
        self._key_credential = (
            AzureKeyCredential(settings.vision_api_key) if settings.vision_api_key else None
        )
        self._aad_credential = None if self._key_credential else DefaultAzureCredential()

    def _headers(self, content_type: str) -> dict:
        headers = {"Content-Type": content_type}
        if self._key_credential:
            headers["Ocp-Apim-Subscription-Key"] = self._key_credential.key
        else:
            token = self._aad_credential.get_token(_TOKEN_SCOPE).token
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def embed_image_bytes(self, image_bytes: bytes) -> List[float]:
        """Vectorize raw image bytes (jpg/png/bmp/gif) -> embedding vector."""
        url = f"{self._endpoint}/computervision/retrieval:vectorizeImage"
        params = {"api-version": self._api_version, "model-version": "2023-04-15"}
        resp = requests.post(
            url,
            params=params,
            headers=self._headers("application/octet-stream"),
            data=image_bytes,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["vector"]

    def embed_image_url(self, image_url: str) -> List[float]:
        """Vectorize an already-hosted image (e.g. a Blob Storage SAS/public URL)."""
        url = f"{self._endpoint}/computervision/retrieval:vectorizeImage"
        params = {"api-version": self._api_version, "model-version": "2023-04-15"}
        resp = requests.post(
            url,
            params=params,
            headers=self._headers("application/json"),
            json={"url": image_url},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["vector"]

    def embed_text(self, text: str) -> List[float]:
        """Vectorize free text into the SAME embedding space as images.

        This is what powers text -> image search: embed the user's query text,
        then do a vector similarity search against stored image vectors.
        """
        url = f"{self._endpoint}/computervision/retrieval:vectorizeText"
        params = {"api-version": self._api_version, "model-version": "2023-04-15"}
        resp = requests.post(
            url,
            params=params,
            headers=self._headers("application/json"),
            json={"text": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["vector"]


embedding_client = MultimodalEmbeddingClient()

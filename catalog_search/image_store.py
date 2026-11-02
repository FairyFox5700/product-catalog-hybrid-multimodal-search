"""Upload and retrieve product images in Azure Blob Storage."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    generate_blob_sas,
)

from .config import settings


class ImageStore:
    """Lazily connects to Blob Storage — no network calls happen until a method
    that actually needs the container is invoked, so importing/constructing this
    class is safe in tests and in environments without live Azure credentials."""

    def __init__(self) -> None:
        self._client: BlobServiceClient | None = None
        self._container = None
        self._container_name = settings.storage_container
        self._container_ensured = False

    def _get_client(self) -> BlobServiceClient:
        if self._client is None:
            if settings.storage_connection_string:
                self._client = BlobServiceClient.from_connection_string(
                    settings.storage_connection_string
                )
            else:
                self._client = BlobServiceClient(
                    account_url=settings.storage_account_url,
                    credential=DefaultAzureCredential(),
                )
        return self._client

    def _get_container(self):
        if self._container is None:
            self._container = self._get_client().get_container_client(self._container_name)
        if not self._container_ensured:
            if not self._container.exists():
                self._container.create_container()
            self._container_ensured = True
        return self._container

    def upload_image(self, local_path: str | Path, blob_name: str | None = None) -> str:
        """Upload a local image file and return its blob name."""
        local_path = Path(local_path)
        blob_name = blob_name or local_path.name
        with open(local_path, "rb") as f:
            self._get_container().upload_blob(name=blob_name, data=f, overwrite=True)
        return blob_name

    def get_url(self, blob_name: str, sas_hours: int = 24 * 7) -> str:
        """Return a time-limited, read-only SAS URL for the image.

        A readable URL is required so Azure AI Vision can download the image to
        compute its multimodal embedding, and so search results can render thumbnails.
        """
        container = self._get_container()
        client = self._get_client()
        blob_client = container.get_blob_client(blob_name)
        account_key = getattr(client.credential, "account_key", None)
        if account_key:
            sas = generate_blob_sas(
                account_name=client.account_name,
                container_name=self._container_name,
                blob_name=blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.now(timezone.utc) + timedelta(hours=sas_hours),
            )
            return f"{blob_client.url}?{sas}"
        # Falls back to user-delegation SAS when using AAD auth (Managed Identity/az login).
        user_delegation_key = client.get_user_delegation_key(
            key_start_time=datetime.now(timezone.utc),
            key_expiry_time=datetime.now(timezone.utc) + timedelta(hours=sas_hours),
        )
        sas = generate_blob_sas(
            account_name=client.account_name,
            container_name=self._container_name,
            blob_name=blob_name,
            user_delegation_key=user_delegation_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=sas_hours),
        )
        return f"{blob_client.url}?{sas}"

    def download_bytes(self, blob_name: str) -> bytes:
        return self._get_container().get_blob_client(blob_name).download_blob().readall()


# Lazily-constructed module singleton (no network calls until first real use).
image_store = ImageStore()

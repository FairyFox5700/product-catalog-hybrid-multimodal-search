"""Centralized configuration loaded from environment variables / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _req(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    # Azure AI Search
    search_endpoint: str = _req("AZURE_SEARCH_ENDPOINT")
    search_index_name: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "product-catalog")
    search_api_key: str = os.getenv("AZURE_SEARCH_API_KEY", "")

    # Azure Blob Storage
    storage_account_url: str = os.getenv("AZURE_STORAGE_ACCOUNT_URL", "")
    storage_container: str = os.getenv("AZURE_STORAGE_CONTAINER", "product-images")
    storage_connection_string: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")

    # Azure AI Vision (multimodal embeddings)
    vision_endpoint: str = _req("AZURE_AI_VISION_ENDPOINT")
    vision_api_key: str = os.getenv("AZURE_AI_VISION_API_KEY", "")
    vision_api_version: str = os.getenv("AZURE_AI_VISION_API_VERSION", "2024-02-01")

    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))


settings = Settings()

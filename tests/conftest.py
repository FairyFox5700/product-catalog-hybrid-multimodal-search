"""Shared pytest fixtures. Sets dummy env vars BEFORE catalog_search.config is
ever imported, so the whole package can be imported/tested without any real
Azure resources or network access."""
import os
import sys
from pathlib import Path

os.environ.setdefault("AZURE_SEARCH_ENDPOINT", "https://dummy-search.search.windows.net")
os.environ.setdefault("AZURE_SEARCH_INDEX_NAME", "product-catalog-test")
os.environ.setdefault("AZURE_SEARCH_API_KEY", "dummy-search-key")
os.environ.setdefault("AZURE_STORAGE_ACCOUNT_URL", "https://dummystorage.blob.core.windows.net")
os.environ.setdefault("AZURE_STORAGE_CONTAINER", "product-images-test")
os.environ.setdefault(
    "AZURE_STORAGE_CONNECTION_STRING",
    "DefaultEndpointsProtocol=https;AccountName=dummystorage;AccountKey=ZHVtbXlrZXk=;EndpointSuffix=core.windows.net",
)
os.environ.setdefault("AZURE_AI_VISION_ENDPOINT", "https://dummy-vision.cognitiveservices.azure.com")
os.environ.setdefault("AZURE_AI_VISION_API_KEY", "dummy-vision-key")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

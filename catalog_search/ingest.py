"""Ingest product catalog items: upload image, compute multimodal embeddings, index doc."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from tqdm import tqdm

from .config import settings
from .embeddings import embedding_client
from .image_store import image_store


@dataclass
class CatalogItem:
    name: str
    description: str
    category: str
    brand: str
    price: float
    image_path: str
    in_stock: bool = True
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


def _get_search_client() -> SearchClient:
    credential = (
        AzureKeyCredential(settings.search_api_key)
        if settings.search_api_key
        else DefaultAzureCredential()
    )
    return SearchClient(
        endpoint=settings.search_endpoint,
        index_name=settings.search_index_name,
        credential=credential,
    )


def load_catalog(json_path: str | Path) -> List[CatalogItem]:
    """Load catalog metadata from a JSON file: a list of objects with
    name/description/category/brand/price/image_path/in_stock."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return [CatalogItem(**item) for item in data]


def ingest_items(items: List[CatalogItem]) -> None:
    search_client = _get_search_client()
    docs = []

    for item in tqdm(items, desc="Ingesting products"):
        blob_name = image_store.upload_image(item.image_path, blob_name=f"{item.id}.jpg")
        image_url = image_store.get_url(blob_name)

        # Image embedding: multimodal vector of the product photo itself.
        image_vector = embedding_client.embed_image_url(image_url)

        # Text embedding: multimodal vector of name+description, in the SAME space as
        # image_vector, so a vector search against image_vector can also be matched
        # against text_vector for text<->text and cross-modal re-ranking scenarios.
        combined_text = f"{item.name}. {item.category}. {item.brand}. {item.description}"
        text_vector = embedding_client.embed_text(combined_text)

        docs.append(
            {
                "id": item.id,
                "name": item.name,
                "description": item.description,
                "category": item.category,
                "brand": item.brand,
                "price": item.price,
                "in_stock": item.in_stock,
                "image_url": image_url,
                "image_blob_name": blob_name,
                "text_vector": text_vector,
                "image_vector": image_vector,
            }
        )

    # Batch upload (chunks of 100-1000 recommended by Azure AI Search).
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        result = search_client.upload_documents(documents=batch)
        failed = [r for r in result if not r.succeeded]
        if failed:
            print(f"WARNING: {len(failed)} documents failed to index: {failed}")

    print(f"Indexed {len(docs)} products into '{settings.search_index_name}'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest product catalog into Azure AI Search")
    parser.add_argument("catalog_json", help="Path to catalog.json describing products + image paths")
    args = parser.parse_args()

    catalog_items = load_catalog(args.catalog_json)
    ingest_items(catalog_items)

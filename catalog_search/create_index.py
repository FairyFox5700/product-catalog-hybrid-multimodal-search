"""Create (or update) the Azure AI Search index used for the product catalog.

The index combines:
  - Plain searchable/filterable text fields  -> classic full-text (BM25) search
  - A `text_vector` field                     -> semantic text embedding
  - An `image_vector` field                   -> multimodal image embedding
    (same embedding space as text queries, enabling text -> image retrieval)
  - A semantic configuration                  -> L2 semantic re-ranking on top of
                                                   the hybrid (BM25 + vector) result set

Together this gives true "hybrid + multimodal" search: a single query can combine
keyword matching, vector similarity over product photos, and semantic re-ranking.
"""
from __future__ import annotations

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

from .config import settings

VECTOR_SEARCH_PROFILE = "catalog-hnsw-profile"
VECTOR_ALGORITHM = "catalog-hnsw-algorithm"
SEMANTIC_CONFIG_NAME = "catalog-semantic-config"


def _get_index_client() -> SearchIndexClient:
    credential = (
        AzureKeyCredential(settings.search_api_key)
        if settings.search_api_key
        else DefaultAzureCredential()
    )
    return SearchIndexClient(endpoint=settings.search_endpoint, credential=credential)


def build_index_definition() -> SearchIndex:
    dims = settings.embedding_dimensions

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="name", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(
            name="description", type=SearchFieldDataType.String, analyzer_name="en.microsoft"
        ),
        SearchableField(
            name="category",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchableField(name="brand", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="price", type=SearchFieldDataType.Double, filterable=True, sortable=True),
        SimpleField(name="in_stock", type=SearchFieldDataType.Boolean, filterable=True),
        SimpleField(name="image_url", type=SearchFieldDataType.String),
        SimpleField(name="image_blob_name", type=SearchFieldDataType.String),
        # --- Vector fields (multimodal) ---
        SearchField(
            name="text_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dims,
            vector_search_profile_name=VECTOR_SEARCH_PROFILE,
        ),
        SearchField(
            name="image_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dims,
            vector_search_profile_name=VECTOR_SEARCH_PROFILE,
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=VECTOR_ALGORITHM,
                parameters=HnswParameters(m=4, ef_construction=400, metric=VectorSearchAlgorithmMetric.COSINE),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_SEARCH_PROFILE,
                algorithm_configuration_name=VECTOR_ALGORITHM,
            )
        ],
    )

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="name"),
                    content_fields=[SemanticField(field_name="description")],
                    keywords_fields=[
                        SemanticField(field_name="category"),
                        SemanticField(field_name="brand"),
                    ],
                ),
            )
        ]
    )

    return SearchIndex(
        name=settings.search_index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def create_or_update_index() -> None:
    client = _get_index_client()
    index = build_index_definition()
    client.create_or_update_index(index)
    print(f"Index '{settings.search_index_name}' created/updated at {settings.search_endpoint}")


if __name__ == "__main__":
    create_or_update_index()

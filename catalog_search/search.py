"""
Hybrid + multimodal search over the product catalog.

Three query modes are supported, all using the same underlying machinery
(BM25 full-text + vector similarity + optional semantic re-ranking):

1. search_by_text(query)       -> hybrid: keyword match on name/description/category/brand
                                   PLUS vector similarity against image_vector using a
                                   text embedding (cross-modal: text finds matching photos)
2. search_by_image(image_bytes)-> pure vector similarity: embed the uploaded photo and
                                   find the closest product images ("find similar items")
3. search_combined(...)        -> keyword + both vectors + filters, fully hybrid
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from .config import settings
from .create_index import SEMANTIC_CONFIG_NAME
from .embeddings import embedding_client


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


_SELECT_FIELDS = ["id", "name", "description", "category", "brand", "price", "in_stock", "image_url"]


def _run_search(
    *,
    search_text: Optional[str],
    vector_queries: List[VectorizedQuery],
    filter_expr: Optional[str],
    top: int,
    use_semantic: bool,
) -> List[Dict[str, Any]]:
    client = _get_search_client()
    kwargs: Dict[str, Any] = dict(
        search_text=search_text or "*",
        vector_queries=vector_queries or None,
        filter=filter_expr,
        select=_SELECT_FIELDS,
        top=top,
    )
    if use_semantic:
        kwargs.update(
            query_type="semantic",
            semantic_configuration_name=SEMANTIC_CONFIG_NAME,
        )
    results = client.search(**kwargs)
    return [dict(r) for r in results]


def search_by_text(
    query: str,
    *,
    top: int = 10,
    filter_expr: Optional[str] = None,
    use_semantic_ranking: bool = True,
) -> List[Dict[str, Any]]:
    """Hybrid text query: BM25 keyword search + vector search over image_vector
    (query text is embedded into the multimodal space to retrieve matching photos),
    optionally re-ranked by the L2 semantic reranker.
    """
    query_vector = embedding_client.embed_text(query)
    vector_query = VectorizedQuery(
        vector=query_vector, k_nearest_neighbors=50, fields="image_vector,text_vector"
    )
    return _run_search(
        search_text=query,
        vector_queries=[vector_query],
        filter_expr=filter_expr,
        top=top,
        use_semantic=use_semantic_ranking,
    )


def search_by_image(
    image_bytes: bytes,
    *,
    top: int = 10,
    filter_expr: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Reverse image search / "find similar products": embed the uploaded photo and
    run pure vector similarity against indexed image vectors."""
    query_vector = embedding_client.embed_image_bytes(image_bytes)
    vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=50, fields="image_vector")
    return _run_search(
        search_text=None,
        vector_queries=[vector_query],
        filter_expr=filter_expr,
        top=top,
        use_semantic=False,  # semantic ranking needs a text query to re-rank against
    )


def search_combined(
    *,
    text: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    top: int = 10,
    filter_expr: Optional[str] = None,
    use_semantic_ranking: bool = True,
) -> List[Dict[str, Any]]:
    """Fully hybrid, multimodal query: optional keyword text + optional text vector +
    optional image vector, all combined by Azure AI Search's Reciprocal Rank Fusion (RRF),
    then (optionally) semantically re-ranked. Use this when a shopper types a description
    AND/OR uploads a reference photo (e.g. "find this but in blue")."""
    vector_queries: List[VectorizedQuery] = []
    if text:
        vector_queries.append(
            VectorizedQuery(
                vector=embedding_client.embed_text(text),
                k_nearest_neighbors=50,
                fields="text_vector,image_vector",
            )
        )
    if image_bytes:
        vector_queries.append(
            VectorizedQuery(
                vector=embedding_client.embed_image_bytes(image_bytes),
                k_nearest_neighbors=50,
                fields="image_vector",
            )
        )
    return _run_search(
        search_text=text,
        vector_queries=vector_queries,
        filter_expr=filter_expr,
        top=top,
        use_semantic=use_semantic_ranking and bool(text),
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Query the product catalog (hybrid + multimodal)")
    parser.add_argument("query", help="Free-text search query")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--filter", dest="filter_expr", default=None, help="OData filter, e.g. \"category eq 'Shoes'\"")
    args = parser.parse_args()

    hits = search_by_text(args.query, top=args.top, filter_expr=args.filter_expr)
    print(json.dumps(hits, indent=2, default=str))

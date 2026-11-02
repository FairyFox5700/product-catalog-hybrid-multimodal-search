"""
Streamlit demo UI for the hybrid + multimodal product catalog search.

Run with:
    streamlit run app.py

Requires a configured .env (see .env.example) pointing at a live Azure AI Search
index (created via `python -m catalog_search.create_index`) that has already
been populated (via `python -m catalog_search.ingest catalog.sample.json`).
"""
from __future__ import annotations

import streamlit as st

from catalog_search.search import search_by_image, search_by_text, search_combined

st.set_page_config(page_title="Product Catalog — Hybrid Multimodal Search", page_icon="🔎", layout="wide")

st.title("🔎 Product Catalog — Hybrid Multimodal Search")
st.caption(
    "Azure AI Search (BM25 + vector + semantic ranking) · Azure AI Vision multimodal "
    "embeddings · Azure Blob Storage for images"
)

with st.sidebar:
    st.header("Query options")
    top_k = st.slider("Number of results", min_value=1, max_value=25, value=8)
    use_semantic = st.checkbox("Use semantic re-ranking", value=True)
    category_filter = st.text_input("Filter: category equals (optional)", value="")
    in_stock_only = st.checkbox("In stock only", value=False)

    filter_clauses = []
    if category_filter.strip():
        escaped = category_filter.replace("'", "''")
        filter_clauses.append(f"category eq '{escaped}'")
    if in_stock_only:
        filter_clauses.append("in_stock eq true")
    filter_expr = " and ".join(filter_clauses) if filter_clauses else None

tab_text, tab_image, tab_combined = st.tabs(
    ["🔤 Search by text", "🖼️ Search by image", "🧩 Combined (text + photo)"]
)


def render_results(results):
    if not results:
        st.info("No results found.")
        return
    cols = st.columns(4)
    for i, hit in enumerate(results):
        col = cols[i % 4]
        with col:
            if hit.get("image_url"):
                st.image(hit["image_url"], use_container_width=True)
            st.markdown(f"**{hit.get('name', 'Unknown')}**")
            st.caption(f"{hit.get('brand', '')} · {hit.get('category', '')}")
            st.write(f"${hit.get('price', 0):.2f}" if hit.get("price") is not None else "")
            stock = "✅ In stock" if hit.get("in_stock") else "❌ Out of stock"
            st.caption(stock)
            score = hit.get("@search.score") or hit.get("@search.rerankerScore")
            if score is not None:
                st.caption(f"score: {score:.3f}")
            st.divider()


with tab_text:
    query = st.text_input("Describe what you're looking for", placeholder="e.g. warm waterproof winter coat")
    if st.button("Search", key="text_search_btn") and query:
        with st.spinner("Searching catalog..."):
            results = search_by_text(
                query, top=top_k, filter_expr=filter_expr, use_semantic_ranking=use_semantic
            )
        render_results(results)

with tab_image:
    uploaded = st.file_uploader("Upload a reference photo", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded and st.button("Find similar products", key="image_search_btn"):
        with st.spinner("Finding visually similar products..."):
            results = search_by_image(uploaded.getvalue(), top=top_k, filter_expr=filter_expr)
        st.image(uploaded, caption="Your reference photo", width=200)
        render_results(results)

with tab_combined:
    combo_text = st.text_input("Refinement text (optional)", placeholder="e.g. but in blue", key="combo_text")
    combo_image = st.file_uploader(
        "Reference photo (optional)", type=["jpg", "jpeg", "png", "bmp"], key="combo_image"
    )
    if st.button("Search", key="combined_search_btn") and (combo_text or combo_image):
        with st.spinner("Running combined hybrid + multimodal search..."):
            results = search_combined(
                text=combo_text or None,
                image_bytes=combo_image.getvalue() if combo_image else None,
                top=top_k,
                filter_expr=filter_expr,
                use_semantic_ranking=use_semantic,
            )
        render_results(results)

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.pipeline import CitationAwareRAG

settings = get_settings()
rag = CitationAwareRAG(settings)

st.set_page_config(page_title=settings.app_title, layout="wide")
st.title(settings.app_title)
st.caption("Ask questions over indexed academic documents and inspect chunk-level citations.")

query = st.text_input("Question", placeholder="What method does the paper propose?")
top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=settings.top_k)

if st.button("Run Query", type="primary"):
    if not query.strip():
        st.warning("Enter a question first.")
    else:
        try:
            result = rag.answer(query, top_k=top_k)
        except FileNotFoundError:
            st.error("Index not found. Run `python scripts/build_index.py` first.")
        else:
            st.subheader("Answer")
            st.text(result.answer)
            st.subheader("Citations")
            for citation in result.citations:
                st.json(citation)

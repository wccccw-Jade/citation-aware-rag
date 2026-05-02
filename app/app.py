from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
REQUEST_TIMEOUT = 30


class APIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _format_datetime(value: str | None) -> str:
    if not value:
        return "-"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.strftime("%Y-%m-%d %H:%M")


def _file_name(path: str | None) -> str:
    return Path(path or "").name or "-"


def _api_request(method: str, path: str, **kwargs: Any) -> Any:
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        raise APIError(f"API connection failed: {exc}") from exc

    if response.status_code >= 400:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        message = error.get("message") or response.text or f"HTTP {response.status_code}"
        raise APIError(message, status_code=response.status_code)

    if not response.content:
        return None
    return response.json()


def _get_health() -> dict[str, Any] | None:
    try:
        return _api_request("GET", "/health")
    except APIError:
        return None


def _list_documents() -> list[dict[str, Any]]:
    payload = _api_request("GET", "/documents")
    return payload.get("documents", [])


def _upload_document(uploaded_file) -> dict[str, Any]:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    return _api_request("POST", "/documents/upload", files=files)


def _delete_document(document_id: str) -> dict[str, Any]:
    return _api_request("DELETE", f"/documents/{document_id}")


def _reindex_document(document_id: str) -> dict[str, Any]:
    return _api_request("POST", f"/documents/{document_id}/reindex")


def _query_documents(query: str, top_k: int) -> dict[str, Any]:
    return _api_request("POST", "/query", json={"query": query, "top_k": top_k})


def _status_badge(status: str) -> str:
    colors = {
        "indexed": "#16794c",
        "processing": "#9a6700",
        "uploaded": "#315f9d",
        "failed": "#b42318",
    }
    color = colors.get(status, "#475467")
    return (
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"background:{color};color:white;font-size:12px'>{status}</span>"
    )


def _render_document_row(document: dict[str, Any]) -> None:
    status_value = document.get("status", "-")
    row = st.container(border=True)
    with row:
        top = st.columns([2.2, 1.1, 1.2, 1.2, 1.4])
        top[0].markdown(f"**{document.get('filename', '-')}**")
        top[1].markdown(_status_badge(status_value), unsafe_allow_html=True)
        top[2].caption(f"Created {_format_datetime(document.get('created_at'))}")
        top[3].caption(f"Indexed {_format_datetime(document.get('indexed_at'))}")
        actions = top[4].columns(2)

        processing = status_value == "processing"
        if actions[0].button("Reindex", key=f"reindex-{document['id']}", disabled=processing):
            try:
                result = _reindex_document(document["id"])
            except APIError as exc:
                st.error(str(exc))
            else:
                st.success(f"Queued task {result['task_id']}")
                st.rerun()

        if actions[1].button("Delete", key=f"delete-{document['id']}", disabled=processing):
            try:
                _delete_document(document["id"])
            except APIError as exc:
                st.error(str(exc))
            else:
                st.success("Deleted")
                st.rerun()

        meta = st.columns([2, 3])
        meta[0].caption(f"ID {document.get('id', '-')}")
        meta[1].caption(f"Source {_file_name(document.get('source_path'))}")
        if document.get("error_message"):
            st.error(document["error_message"])


def _render_documents_tab() -> None:
    upload = st.file_uploader("Upload document", type=["pdf", "txt", "md"])
    if st.button("Upload and Index", type="primary", disabled=upload is None):
        try:
            result = _upload_document(upload)
        except APIError as exc:
            st.error(str(exc))
        else:
            st.success(f"Queued task {result['task_id']}")
            st.rerun()

    st.divider()
    header = st.columns([1, 1])
    header[0].subheader("Documents")
    if header[1].button("Refresh", use_container_width=True):
        st.rerun()

    try:
        documents = _list_documents()
    except APIError as exc:
        st.error(str(exc))
        return

    if not documents:
        st.info("No documents.")
        return

    for document in documents:
        _render_document_row(document)


def _chunk_by_label(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    chunks: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(result.get("retrieved_chunks", []), start=1):
        chunks[index] = item
    return chunks


def _render_citations(result: dict[str, Any]) -> None:
    citations = result.get("citations", [])
    if not citations:
        st.info("No citations returned.")
        return

    chunks = _chunk_by_label(result)
    for citation in citations:
        label = int(citation.get("label", 0))
        chunk_item = chunks.get(label, {})
        chunk = chunk_item.get("chunk", {})
        score = citation.get("score", chunk_item.get("score"))
        title = citation.get("title") or chunk.get("title") or _file_name(citation.get("source_path"))
        page = citation.get("page_number") or chunk.get("page_number")
        suffix = f"page {page}" if page else "source"
        with st.expander(f"[{label}] {title} · {suffix} · score {score}"):
            cols = st.columns([1.4, 1.1, 2.2])
            cols[0].metric("Label", f"[{label}]")
            cols[1].metric("Score", f"{score}")
            cols[2].caption(citation.get("source_path") or chunk.get("source_path") or "-")
            st.write(chunk.get("text") or "No preview text returned.")


def _render_query_tab() -> None:
    query = st.text_area("Question", placeholder="What methodology does the paper use?", height=110)
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=min(max(DEFAULT_TOP_K, 1), 10))

    if st.button("Ask", type="primary", disabled=not query.strip()):
        try:
            st.session_state["last_answer"] = _query_documents(query.strip(), top_k)
        except APIError as exc:
            st.error(str(exc))

    result = st.session_state.get("last_answer")
    if not result:
        return

    st.subheader("Answer")
    st.write(result.get("answer", ""))
    details = st.columns(3)
    details[0].metric("Confidence", result.get("confidence") or "-")
    details[1].metric("Mode", result.get("generation_mode") or "-")
    validation = result.get("citation_validation", {})
    details[2].metric("Citations Valid", "Yes" if validation.get("valid") else "No")
    if result.get("limitations"):
        st.warning(result["limitations"])

    st.subheader("Citations")
    _render_citations(result)


def _render_sidebar() -> None:
    st.sidebar.caption(API_BASE_URL)
    health = _get_health()
    if health:
        st.sidebar.success("API online")
    else:
        st.sidebar.error("API unavailable")


def main() -> None:
    st.set_page_config(page_title="Citation-Aware RAG", layout="wide")
    st.title("Citation-Aware RAG")
    _render_sidebar()

    documents_tab, query_tab = st.tabs(["Documents", "Ask"])
    with documents_tab:
        _render_documents_tab()
    with query_tab:
        _render_query_tab()


if __name__ == "__main__":
    main()

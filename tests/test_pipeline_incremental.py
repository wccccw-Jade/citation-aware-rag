from pathlib import Path

from src.config import Settings
from src.pipeline import CitationAwareRAG
from src.utils import read_jsonl
from src.vector_store import VectorStore


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        index_dir=tmp_path / "index",
        embedding_provider="local",
        vector_store_provider="numpy",
        use_faiss=False,
    )


def test_index_document_appends_only_that_document(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.raw_data_dir.mkdir()
    first = settings.raw_data_dir / "first.txt"
    second = settings.raw_data_dir / "second.txt"
    first.write_text("First document about retrieval.", encoding="utf-8")
    second.write_text("Second document about citations.", encoding="utf-8")

    rag = CitationAwareRAG(settings)
    assert rag.index_document(first) == {"documents": 1, "chunks": 1}
    assert rag.index_document(second) == {"documents": 1, "chunks": 1}
    assert rag.index_document(second) == {"documents": 0, "chunks": 0}

    documents = read_jsonl(settings.processed_data_dir / "documents.jsonl")
    chunks = read_jsonl(settings.processed_data_dir / "chunks.jsonl")
    store = VectorStore(settings.index_dir, use_faiss=False)
    store.load()

    assert [Path(row["source_path"]).name for row in documents] == ["first.txt", "second.txt"]
    assert [Path(row["source_path"]).name for row in chunks] == ["first.txt", "second.txt"]
    assert [Path(row["source_path"]).name for row in store.metadata] == ["first.txt", "second.txt"]


def test_reindex_document_replaces_existing_document_rows(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.raw_data_dir.mkdir()
    source = settings.raw_data_dir / "notes.txt"
    source.write_text("Original retrieval note.", encoding="utf-8")

    rag = CitationAwareRAG(settings)
    assert rag.index_document(source) == {"documents": 1, "chunks": 1}

    source.write_text("Updated citation note.", encoding="utf-8")
    assert rag.reindex_document(source) == {"documents": 1, "chunks": 1}

    documents = read_jsonl(settings.processed_data_dir / "documents.jsonl")
    chunks = read_jsonl(settings.processed_data_dir / "chunks.jsonl")
    store = VectorStore(settings.index_dir, use_faiss=False)
    store.load()

    assert len(documents) == 1
    assert len(chunks) == 1
    assert documents[0]["text"] == "Updated citation note."
    assert chunks[0]["text"] == "Updated citation note."
    assert len(store.metadata) == 1

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .chunking import chunk_documents
from .config import Settings, get_settings
from .embedding import EmbeddingProvider, create_embedding_model
from .generator import generate_answer
from .preprocess import load_document, load_documents, merge_persisted_documents, persist_documents
from .preprocess import remove_persisted_documents_by_source
from .retrieval import Retriever
from .schemas import AnswerResult, ChunkRecord
from .utils import read_jsonl, write_jsonl
from .vector_store import VectorStoreBackend, create_vector_store


class CitationAwareRAG:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.embedder: EmbeddingProvider = create_embedding_model(self.settings)
        self.vector_store: VectorStoreBackend = create_vector_store(
            self.settings.index_dir,
            self.settings.vector_store_provider,
            use_faiss=self.settings.use_faiss,
        )
        self.chunks: list[ChunkRecord] = []
        self._chunks_mtime: float | None = None

    def build_index(self) -> dict[str, int]:
        documents = load_documents(self.settings.raw_data_dir)
        persist_documents(documents, self.settings.processed_data_dir)

        chunks = chunk_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        write_jsonl(
            self.settings.processed_data_dir / "chunks.jsonl",
            [chunk.model_dump() for chunk in chunks],
        )

        embeddings = self.embedder.encode([self._chunk_search_text(chunk) for chunk in chunks])
        self.vector_store.build(
            embeddings,
            [chunk.model_dump() for chunk in chunks],
            index_config=self._index_config(),
        )
        self.chunks = chunks
        self._chunks_mtime = (self.settings.processed_data_dir / "chunks.jsonl").stat().st_mtime
        return {"documents": len(documents), "chunks": len(chunks)}

    def index_document(self, document_path: Path) -> dict[str, int]:
        documents = load_document(document_path)
        document_count = merge_persisted_documents(documents, self.settings.processed_data_dir)

        chunks = chunk_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        chunk_path = self.settings.processed_data_dir / "chunks.jsonl"
        existing_chunks = read_jsonl(chunk_path) if chunk_path.exists() else []
        existing_chunk_ids = {row["chunk_id"] for row in existing_chunks}
        chunk_rows = [chunk.model_dump() for chunk in chunks]
        new_chunk_rows = [row for row in chunk_rows if row["chunk_id"] not in existing_chunk_ids]
        if not chunks:
            self.chunks = [ChunkRecord(**row) for row in existing_chunks]
            self._chunks_mtime = chunk_path.stat().st_mtime if chunk_path.exists() else None
            return {"documents": document_count, "chunks": 0}

        embeddings = self.embedder.encode([self._chunk_search_text(chunk) for chunk in chunks])
        chunk_count = self.vector_store.append(
            embeddings,
            chunk_rows,
            index_config=self._index_config(),
        )

        if not new_chunk_rows:
            self.chunks = [ChunkRecord(**row) for row in existing_chunks]
            self._chunks_mtime = chunk_path.stat().st_mtime if chunk_path.exists() else None
            return {"documents": document_count, "chunks": chunk_count}

        merged_chunks = [*existing_chunks, *new_chunk_rows]
        write_jsonl(chunk_path, merged_chunks)

        self.chunks = [ChunkRecord(**row) for row in merged_chunks]
        self._chunks_mtime = chunk_path.stat().st_mtime
        return {"documents": document_count, "chunks": chunk_count}

    def reindex_document(self, document_path: Path) -> dict[str, int]:
        self.delete_document_index(str(document_path))
        return self.index_document(document_path)

    def delete_document_index(self, source_path: str) -> dict[str, int]:
        document_stats = remove_persisted_documents_by_source(source_path, self.settings.processed_data_dir)
        chunk_path = self.settings.processed_data_dir / "chunks.jsonl"
        existing_chunks = read_jsonl(chunk_path) if chunk_path.exists() else []
        remaining_chunks = [row for row in existing_chunks if row["source_path"] != source_path]
        removed_chunks = len(existing_chunks) - len(remaining_chunks)
        write_jsonl(chunk_path, remaining_chunks)
        removed_vectors = self.vector_store.delete_by_source_path(source_path)
        self.chunks = [ChunkRecord(**row) for row in remaining_chunks]
        self._chunks_mtime = chunk_path.stat().st_mtime
        return {
            "documents": document_stats["documents"],
            "chunks": removed_chunks,
            "vectors": removed_vectors,
        }

    def load_index(self) -> None:
        chunk_path = self.settings.processed_data_dir / "chunks.jsonl"
        if not chunk_path.exists():
            raise FileNotFoundError("Missing processed chunks. Run scripts/build_index.py first.")
        self.chunks = [ChunkRecord(**row) for row in read_jsonl(chunk_path)]
        self.vector_store.load()
        self._chunks_mtime = chunk_path.stat().st_mtime

    def answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> AnswerResult:
        if not self.chunks or self._index_files_changed():
            self.load_index()
        retriever = Retriever(self.embedder, self.vector_store, self.settings)
        retrieved = retriever.retrieve(query, self.chunks, top_k or self.settings.top_k, metadata_filter=metadata_filter)
        return generate_answer(query, retrieved, settings=self.settings)

    @staticmethod
    def _chunk_search_text(chunk: ChunkRecord) -> str:
        return f"{chunk.title} {Path(chunk.source_path).name} {chunk.text}"

    def _index_config(self) -> dict[str, Any]:
        return {
            "embedding_provider": self.embedder.provider_name,
            "embedding_model_name": self.embedder.model_name,
            "embedding_dim": self.embedder.dim,
            "vector_store_provider": self.settings.vector_store_provider,
        }

    def _index_files_changed(self) -> bool:
        chunk_path = self.settings.processed_data_dir / "chunks.jsonl"
        if not chunk_path.exists() or self._chunks_mtime is None:
            return False
        return chunk_path.stat().st_mtime > self._chunks_mtime

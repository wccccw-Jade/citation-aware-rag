from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .chunking import chunk_documents
from .config import Settings, get_settings
from .embedding import EmbeddingProvider, create_embedding_model
from .generator import generate_answer
from .preprocess import load_documents, persist_documents
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
            index_config={
                "embedding_provider": self.embedder.provider_name,
                "embedding_model_name": self.embedder.model_name,
                "embedding_dim": self.embedder.dim,
                "vector_store_provider": self.settings.vector_store_provider,
            },
        )
        self.chunks = chunks
        return {"documents": len(documents), "chunks": len(chunks)}

    def load_index(self) -> None:
        chunk_path = self.settings.processed_data_dir / "chunks.jsonl"
        if not chunk_path.exists():
            raise FileNotFoundError("Missing processed chunks. Run scripts/build_index.py first.")
        self.chunks = [ChunkRecord(**row) for row in read_jsonl(chunk_path)]
        self.vector_store.load()

    def answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> AnswerResult:
        if not self.chunks:
            self.load_index()
        retriever = Retriever(self.embedder, self.vector_store, self.settings)
        retrieved = retriever.retrieve(query, self.chunks, top_k or self.settings.top_k, metadata_filter=metadata_filter)
        return generate_answer(query, retrieved, settings=self.settings)

    @staticmethod
    def _chunk_search_text(chunk: ChunkRecord) -> str:
        return f"{chunk.title} {Path(chunk.source_path).name} {chunk.text}"

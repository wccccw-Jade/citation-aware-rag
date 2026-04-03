from __future__ import annotations

from pathlib import Path
from typing import Optional

from .chunking import chunk_documents
from .config import Settings, get_settings
from .embedding import LocalEmbeddingModel
from .generator import generate_answer
from .preprocess import load_documents, persist_documents
from .retrieval import Retriever
from .schemas import AnswerResult, ChunkRecord
from .utils import read_jsonl, write_jsonl
from .vector_store import VectorStore


class CitationAwareRAG:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = LocalEmbeddingModel(dim=self.settings.embedding_dim)
        self.vector_store = VectorStore(self.settings.index_dir, use_faiss=self.settings.use_faiss)
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
        self.vector_store.build(embeddings, [chunk.model_dump() for chunk in chunks])
        self.chunks = chunks
        return {"documents": len(documents), "chunks": len(chunks)}

    def load_index(self) -> None:
        chunk_path = self.settings.processed_data_dir / "chunks.jsonl"
        if not chunk_path.exists():
            raise FileNotFoundError("Missing processed chunks. Run scripts/build_index.py first.")
        self.chunks = [ChunkRecord(**row) for row in read_jsonl(chunk_path)]
        self.vector_store.load()

    def answer(self, query: str, top_k: Optional[int] = None) -> AnswerResult:
        if not self.chunks:
            self.load_index()
        retriever = Retriever(self.embedder, self.vector_store, self.settings)
        retrieved = retriever.retrieve(query, self.chunks, top_k or self.settings.top_k)
        return generate_answer(query, retrieved)

    @staticmethod
    def _chunk_search_text(chunk: ChunkRecord) -> str:
        return f"{chunk.title} {Path(chunk.source_path).name} {chunk.text}"

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

from .utils import ensure_dir


class VectorStore:
    def __init__(self, index_dir: Path, use_faiss: bool = True) -> None:
        self.index_dir = index_dir
        self.use_faiss = use_faiss and faiss is not None
        self.index: Any = None
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: list[dict] = []

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        ensure_dir(self.index_dir)
        self.metadata = metadata
        self.embeddings = embeddings.astype(np.float32)
        if self.use_faiss:
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(self.embeddings)
            faiss.write_index(index, str(self.index_dir / "index.faiss"))
            self.index = index
        with (self.index_dir / "index_meta.pkl").open("wb") as handle:
            pickle.dump({"metadata": metadata, "embeddings": self.embeddings}, handle)

    def load(self) -> None:
        meta_path = self.index_dir / "index_meta.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing index metadata: {meta_path}")
        with meta_path.open("rb") as handle:
            payload = pickle.load(handle)
        self.metadata = payload["metadata"]
        self.embeddings = payload.get("embeddings")
        faiss_path = self.index_dir / "index.faiss"
        if self.use_faiss and faiss_path.exists():
            self.index = faiss.read_index(str(faiss_path))

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        query = query_embedding.astype(np.float32)
        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(query, top_k)
            return [
                (int(idx), float(score))
                for idx, score in zip(indices[0].tolist(), scores[0].tolist())
                if idx >= 0
            ]
        if self.embeddings is None:
            raise RuntimeError("No embeddings loaded for search.")
        scores = self.embeddings @ query[0]
        best = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in best.tolist()]

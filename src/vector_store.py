from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

from .utils import ensure_dir


class VectorStoreBackend(Protocol):
    metadata: list[dict]

    def build(self, embeddings: np.ndarray, metadata: list[dict], index_config: dict[str, Any] | None = None) -> None:
        """Persist embeddings and metadata."""

    def load(self) -> None:
        """Load persisted index state."""

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[int, float]]:
        """Return matching metadata row indices and similarity scores."""


def _matches_filter(metadata: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
    if not metadata_filter:
        return True
    for key, expected in metadata_filter.items():
        actual = metadata.get(key)
        if actual is None and isinstance(metadata.get("metadata"), dict):
            actual = metadata["metadata"].get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


class VectorStore:
    def __init__(self, index_dir: Path, use_faiss: bool = True) -> None:
        self.index_dir = index_dir
        self.use_faiss = use_faiss and self._faiss_available()
        self.index: Any = None
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: list[dict] = []
        self.index_config: dict[str, Any] = {}

    @staticmethod
    def _load_faiss() -> Any:
        try:
            import faiss
        except ImportError:  # pragma: no cover
            return None
        return faiss

    @classmethod
    def _faiss_available(cls) -> bool:
        return cls._load_faiss() is not None

    def build(self, embeddings: np.ndarray, metadata: list[dict], index_config: dict[str, Any] | None = None) -> None:
        ensure_dir(self.index_dir)
        self.metadata = metadata
        self.embeddings = embeddings.astype(np.float32)
        self.index_config = index_config or {}
        if self.use_faiss:
            faiss = self._load_faiss()
            if faiss is None:
                self.use_faiss = False
            else:
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(self.embeddings)
                faiss.write_index(index, str(self.index_dir / "index.faiss"))
                self.index = index
        with (self.index_dir / "index_meta.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "metadata": metadata,
                    "embeddings": self.embeddings,
                    "index_config": self.index_config,
                },
                handle,
            )

    def load(self) -> None:
        meta_path = self.index_dir / "index_meta.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing index metadata: {meta_path}")
        with meta_path.open("rb") as handle:
            payload = pickle.load(handle)
        self.metadata = payload["metadata"]
        self.embeddings = payload.get("embeddings")
        self.index_config = payload.get("index_config", {})
        faiss_path = self.index_dir / "index.faiss"
        if self.use_faiss and faiss_path.exists():
            faiss = self._load_faiss()
            if faiss is None:
                self.use_faiss = False
            else:
                self.index = faiss.read_index(str(faiss_path))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[int, float]]:
        query = query_embedding.astype(np.float32)
        if self.use_faiss and self.index is not None and not metadata_filter:
            scores, indices = self.index.search(query, top_k)
            return [
                (int(idx), float(score))
                for idx, score in zip(indices[0].tolist(), scores[0].tolist())
                if idx >= 0
            ]
        if self.embeddings is None:
            raise RuntimeError("No embeddings loaded for search.")
        scores = self.embeddings @ query[0]
        best = np.argsort(scores)[::-1]
        results: list[tuple[int, float]] = []
        for idx in best.tolist():
            if not _matches_filter(self.metadata[idx], metadata_filter):
                continue
            results.append((int(idx), float(scores[idx])))
            if len(results) == top_k:
                break
        return results


def create_vector_store(index_dir: Path, provider: str, use_faiss: bool = True) -> VectorStoreBackend:
    normalized = provider.lower().strip()
    if normalized in {"faiss", "local", "numpy"}:
        return VectorStore(index_dir, use_faiss=use_faiss and normalized != "numpy")
    raise ValueError(f"Unsupported vector store provider: {provider}")

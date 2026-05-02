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

    def append(self, embeddings: np.ndarray, metadata: list[dict], index_config: dict[str, Any] | None = None) -> int:
        """Append embeddings and metadata, returning the number of new rows persisted."""

    def delete_by_source_path(self, source_path: str) -> int:
        """Delete rows for a source path, returning the number removed."""

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
        self._rebuild_runtime_index()
        self._persist()

    def append(self, embeddings: np.ndarray, metadata: list[dict], index_config: dict[str, Any] | None = None) -> int:
        if len(metadata) == 0:
            return 0
        ensure_dir(self.index_dir)
        try:
            self.load()
        except FileNotFoundError:
            self.build(embeddings, metadata, index_config=index_config)
            return len(metadata)

        self._validate_append_config(index_config or {})
        existing_chunk_ids = {row.get("chunk_id") for row in self.metadata}
        new_rows: list[tuple[np.ndarray, dict]] = [
            (embedding, row)
            for embedding, row in zip(embeddings.astype(np.float32), metadata)
            if row.get("chunk_id") not in existing_chunk_ids
        ]
        if not new_rows:
            return 0

        new_embeddings = np.array([row[0] for row in new_rows], dtype=np.float32)
        new_metadata = [row[1] for row in new_rows]
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings]).astype(np.float32)
        self.metadata.extend(new_metadata)
        if not self.index_config:
            self.index_config = index_config or {}
        self._rebuild_runtime_index()
        self._persist()
        return len(new_metadata)

    def delete_by_source_path(self, source_path: str) -> int:
        try:
            self.load()
        except FileNotFoundError:
            return 0

        keep_indices = [index for index, row in enumerate(self.metadata) if row.get("source_path") != source_path]
        removed_count = len(self.metadata) - len(keep_indices)
        if removed_count == 0:
            return 0

        self.metadata = [self.metadata[index] for index in keep_indices]
        if self.embeddings is not None:
            if keep_indices:
                self.embeddings = self.embeddings[keep_indices].astype(np.float32)
            else:
                self.embeddings = None
        self.index = None
        if self.embeddings is None:
            (self.index_dir / "index.faiss").unlink(missing_ok=True)
        else:
            self._rebuild_runtime_index()
        self._persist()
        return removed_count

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

    def _validate_append_config(self, index_config: dict[str, Any]) -> None:
        if not self.index_config or not index_config:
            return
        checked_keys = {"embedding_provider", "embedding_model_name", "embedding_dim", "vector_store_provider"}
        for key in checked_keys:
            if self.index_config.get(key) != index_config.get(key):
                raise ValueError(
                    f"Cannot append to index with different {key}: "
                    f"{self.index_config.get(key)!r} != {index_config.get(key)!r}"
                )

    def _rebuild_runtime_index(self) -> None:
        if not self.use_faiss or self.embeddings is None:
            return
        faiss = self._load_faiss()
        if faiss is None:
            self.use_faiss = False
            return
        index = faiss.IndexFlatIP(self.embeddings.shape[1])
        index.add(self.embeddings)
        faiss.write_index(index, str(self.index_dir / "index.faiss"))
        self.index = index

    def _persist(self) -> None:
        with (self.index_dir / "index_meta.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "metadata": self.metadata,
                    "embeddings": self.embeddings,
                    "index_config": self.index_config,
                },
                handle,
            )

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

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import numpy as np

from .config import Settings

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)?")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one normalized embedding per input text."""


class LocalEmbeddingModel:
    provider_name = "local"
    model_name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _iter_features(self, text: str) -> list[str]:
        tokens = _tokenize(text)
        features = list(tokens)
        features.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
        return features

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._iter_features(text):
                digest = hashlib.md5(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                weight = 1.5 if "_" in feature else 1.0
                matrix[row, index] += sign * weight
            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm
        return matrix


class SentenceTransformerEmbeddingModel:
    provider_name = "sentence-transformers"

    def __init__(self, model_name: str, dim: int | None = None) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError(
                "sentence-transformers is not installed. Install optional production "
                "dependencies or set EMBEDDING_PROVIDER=local."
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = dim or int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)


def create_embedding_model(settings: Settings) -> EmbeddingProvider:
    provider = settings.embedding_provider.lower().strip()
    if provider in {"local", "hash", "hashing"}:
        return LocalEmbeddingModel(dim=settings.embedding_dim)
    if provider in {"sentence-transformers", "sentence_transformers", "sbert"}:
        return SentenceTransformerEmbeddingModel(settings.embedding_model_name)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")

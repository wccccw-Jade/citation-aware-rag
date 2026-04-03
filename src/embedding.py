from __future__ import annotations

import hashlib
import re

import numpy as np

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)?")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class LocalEmbeddingModel:
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

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings
from .embedding import EmbeddingProvider
from .schemas import ChunkRecord, RetrievedChunk
from .vector_store import VectorStoreBackend, _matches_filter

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)?")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lower = min(values)
    upper = max(values)
    if math.isclose(lower, upper):
        return {index: 1.0 for index in scores}
    return {index: (score - lower) / (upper - lower) for index, score in scores.items()}


class Retriever:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStoreBackend,
        settings: Settings,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.settings = settings

    def _search_text(self, chunk: ChunkRecord) -> str:
        file_name = Path(chunk.source_path).name
        return f"{chunk.title} {file_name} {chunk.text}"

    def _lexical_scores(self, query: str, chunks: list[ChunkRecord]) -> dict[int, float]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return {}

        query_counts = Counter(query_tokens)
        doc_frequencies: Counter[str] = Counter()
        tokenized_chunks: list[list[str]] = []
        average_length = 0.0
        for chunk in chunks:
            tokens = _tokenize(self._search_text(chunk))
            tokenized_chunks.append(tokens)
            average_length += len(tokens)
            doc_frequencies.update(set(tokens))

        average_length = average_length / max(len(tokenized_chunks), 1)
        scores: dict[int, float] = {}
        total_docs = len(tokenized_chunks)
        for index, tokens in enumerate(tokenized_chunks):
            if not tokens:
                continue
            counts = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for token, query_count in query_counts.items():
                if token not in counts:
                    continue
                doc_freq = doc_frequencies[token]
                idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                tf = counts[token]
                denominator = tf + 1.2 * (1 - 0.75 + 0.75 * (length / max(average_length, 1.0)))
                score += query_count * idf * ((tf * (1.2 + 1)) / denominator)

            if score > 0:
                coverage = len(set(query_tokens) & set(tokens)) / len(set(query_tokens))
                scores[index] = score + 0.5 * coverage
        return scores

    def _mmr_select(
        self,
        ranked: list[tuple[int, float]],
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[tuple[int, float]]:
        embeddings = getattr(self.vector_store, "embeddings", None)
        if embeddings is None or len(ranked) <= top_k:
            return ranked[:top_k]

        candidate_indices = [index for index, _ in ranked]
        score_by_index = dict(ranked)
        selected: list[int] = []
        remaining = candidate_indices[:]

        while remaining and len(selected) < top_k:
            best_index: int | None = None
            best_score = float("-inf")
            for index in remaining:
                relevance = score_by_index[index]
                if selected:
                    diversity_penalty = max(float(embeddings[index] @ embeddings[selected_index]) for selected_index in selected)
                else:
                    diversity_penalty = 0.0
                mmr_score = self.settings.mmr_lambda * relevance - (1 - self.settings.mmr_lambda) * diversity_penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = index
            if best_index is None:
                break
            selected.append(best_index)
            remaining.remove(best_index)

        return [(index, score_by_index[index]) for index in selected]

    def retrieve(
        self,
        query: str,
        chunks: list[ChunkRecord],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedder.encode([query])
        candidate_pool = max(top_k, self.settings.retrieval_candidate_pool)
        dense_results = self.vector_store.search(query_embedding, top_k=candidate_pool, metadata_filter=metadata_filter)
        dense_scores = {index: score for index, score in dense_results}
        filtered_chunks = chunks
        filtered_index_map = list(range(len(chunks)))
        if metadata_filter:
            vector_metadata = getattr(self.vector_store, "metadata", [])
            allowed = {index for index, _ in dense_results}
            if vector_metadata:
                allowed.update(
                    index
                    for index, metadata in enumerate(vector_metadata)
                    if _matches_filter(metadata, metadata_filter)
                )
            filtered_index_map = sorted(allowed)
            filtered_chunks = [chunks[index] for index in filtered_index_map]

        lexical_scores_local = self._lexical_scores(query, filtered_chunks)
        lexical_scores = {filtered_index_map[index]: score for index, score in lexical_scores_local.items()}

        ranked_lexical = sorted(lexical_scores.items(), key=lambda item: item[1], reverse=True)[:candidate_pool]
        candidate_indices = {index for index, _ in dense_results}
        candidate_indices.update(index for index, _ in ranked_lexical)

        normalized_dense = _min_max_normalize({index: dense_scores[index] for index in candidate_indices if index in dense_scores})
        normalized_lexical = _min_max_normalize({index: lexical_scores[index] for index in candidate_indices if index in lexical_scores})

        combined: list[tuple[int, float]] = []
        for index in candidate_indices:
            score = (
                self.settings.dense_weight * normalized_dense.get(index, 0.0)
                + self.settings.lexical_weight * normalized_lexical.get(index, 0.0)
            )
            if score > 0:
                combined.append((index, score))

        combined.sort(key=lambda item: item[1], reverse=True)
        combined = self._mmr_select(combined, query_embedding, top_k)
        return [
            RetrievedChunk(chunk=chunks[index], score=score)
            for index, score in combined
        ]

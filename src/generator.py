from __future__ import annotations

import re
from collections import Counter

from .schemas import AnswerResult, RetrievedChunk

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)?")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _best_evidence_sentences(query: str, retrieved_chunks: list[RetrievedChunk]) -> list[tuple[int, str]]:
    query_tokens = Counter(_tokenize(query))
    evidence: list[tuple[float, int, str]] = []
    for label, item in enumerate(retrieved_chunks, start=1):
        sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(item.chunk.text) if part.strip()]
        for sentence in sentences:
            sentence_tokens = set(_tokenize(sentence))
            if not sentence_tokens:
                continue
            overlap = sum(weight for token, weight in query_tokens.items() if token in sentence_tokens)
            if overlap == 0:
                continue
            score = overlap + item.score
            evidence.append((score, label, sentence))
    evidence.sort(key=lambda row: row[0], reverse=True)

    selected: list[tuple[int, str]] = []
    seen = set()
    for _, label, sentence in evidence:
        normalized = " ".join(sentence.split())
        if normalized in seen:
            continue
        selected.append((label, normalized))
        seen.add(normalized)
        if len(selected) == 3:
            break
    return selected


def generate_answer(query: str, retrieved_chunks: list[RetrievedChunk]) -> AnswerResult:
    if not retrieved_chunks:
        return AnswerResult(query=query, answer="No grounded evidence was found.", citations=[], retrieved_chunks=[])

    citations: list[dict] = []
    for idx, item in enumerate(retrieved_chunks, start=1):
        chunk = item.chunk
        citations.append(
            {
                "label": idx,
                "title": chunk.title,
                "source_path": chunk.source_path,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "score": round(item.score, 4),
            }
        )

    evidence_sentences = _best_evidence_sentences(query, retrieved_chunks)
    if evidence_sentences:
        answer_lines = [f"[{label}] {sentence}" for label, sentence in evidence_sentences]
        answer = "Grounded answer:\n" + "\n".join(answer_lines)
    else:
        answer = "Grounded answer:\nNo grounded answer could be composed from the retrieved evidence."

    return AnswerResult(
        query=query,
        answer=answer,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
    )

from __future__ import annotations

import re
from collections import Counter

from .citation_validator import validate_citations
from .config import Settings
from .llm import LLMProvider, create_llm_provider
from .prompt import build_citation_aware_prompts
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


def _build_citations(
    retrieved_chunks: list[RetrievedChunk],
    generation_mode: str,
    llm_error: str | None = None,
) -> list[dict]:
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
                "generation_mode": generation_mode,
                "llm_error": llm_error,
            }
        )
    return citations


def _generate_fallback_answer(query: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    evidence_sentences = _best_evidence_sentences(query, retrieved_chunks)
    if evidence_sentences:
        answer_lines = [f"[{label}] {sentence}" for label, sentence in evidence_sentences]
        return "Grounded answer:\n" + "\n".join(answer_lines)
    return "Grounded answer:\nNo grounded answer could be composed from the retrieved evidence."


def generate_answer(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    settings: Settings | None = None,
    llm_provider: LLMProvider | None = None,
) -> AnswerResult:
    if not retrieved_chunks:
        answer = "I don't know based on the provided documents."
        validation = validate_citations(answer, [])
        return AnswerResult(
            query=query,
            answer=answer,
            citations=[],
            retrieved_chunks=[],
            citation_validation=validation,
            confidence="low",
            limitations="No retrieved evidence was available.",
            generation_mode="refusal",
        )

    provider = llm_provider
    if provider is None and settings is not None:
        provider = create_llm_provider(settings)

    generation_mode = "extractive"
    llm_error = None
    if provider is not None:
        try:
            system_prompt, user_prompt = build_citation_aware_prompts(query, retrieved_chunks)
            answer = provider.generate(system_prompt, user_prompt)
            generation_mode = provider.provider_name
        except Exception as exc:
            answer = _generate_fallback_answer(query, retrieved_chunks)
            generation_mode = "extractive_fallback_after_llm_error"
            llm_error = f"{type(exc).__name__}: {exc}"
    else:
        answer = _generate_fallback_answer(query, retrieved_chunks)

    citations = _build_citations(retrieved_chunks, generation_mode, llm_error)
    validation = validate_citations(answer, retrieved_chunks)
    confidence = "medium" if validation["valid"] else "low"
    limitations = None
    if validation["reasons"]:
        limitations = "; ".join(validation["reasons"])
    elif llm_error:
        limitations = "LLM generation failed; used extractive fallback."

    return AnswerResult(
        query=query,
        answer=answer,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
        citation_validation=validation,
        confidence=confidence,
        limitations=limitations,
        generation_mode=generation_mode,
    )

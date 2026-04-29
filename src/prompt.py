from __future__ import annotations

from .schemas import RetrievedChunk


def build_context(retrieved_chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for idx, item in enumerate(retrieved_chunks, start=1):
        page = item.chunk.page_number or "?"
        lines.append(
            f"[{idx}] {item.chunk.title} | page={page} | score={item.score:.3f}\n{item.chunk.text}"
        )
    return "\n\n".join(lines)


def build_citation_aware_prompts(query: str, retrieved_chunks: list[RetrievedChunk]) -> tuple[str, str]:
    system_prompt = """You are a citation-aware RAG answer generator.

Rules:
- Answer only using the retrieved context provided by the user.
- Cite every factual claim with retrieval labels like [1] or [2].
- Use only labels that appear in the provided context.
- If the context does not contain enough evidence, say: "I don't know based on the provided documents."
- Do not use external knowledge.
- Keep the answer concise and directly relevant to the question.
"""
    user_prompt = f"""Question:
{query}

Retrieved context:
{build_context(retrieved_chunks)}

Write a grounded answer with citations."""
    return system_prompt, user_prompt

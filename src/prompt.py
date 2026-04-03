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

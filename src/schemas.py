from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    doc_id: str
    source_path: str
    title: str
    page_number: Optional[int] = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    text: str
    page_number: Optional[int] = None
    start_char: int
    end_char: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: ChunkRecord
    score: float


class AnswerResult(BaseModel):
    query: str
    answer: str
    citations: list[dict[str, Any]]
    retrieved_chunks: list[RetrievedChunk]

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ProjectBaseModel(BaseModel):
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        parent_model_dump = getattr(super(), "model_dump", None)
        if parent_model_dump is not None:
            return parent_model_dump(**kwargs)
        return self.dict(**kwargs)


class DocumentRecord(ProjectBaseModel):
    doc_id: str
    source_path: str
    title: str
    page_number: Optional[int] = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(ProjectBaseModel):
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    text: str
    page_number: Optional[int] = None
    start_char: int
    end_char: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(ProjectBaseModel):
    chunk: ChunkRecord
    score: float


class AnswerResult(ProjectBaseModel):
    query: str
    answer: str
    citations: list[dict[str, Any]]
    retrieved_chunks: list[RetrievedChunk]
    citation_validation: dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[str] = None
    limitations: Optional[str] = None
    generation_mode: str = "extractive"

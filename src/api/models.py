from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class UploadResponse(BaseModel):
    document_id: str
    task_id: str
    filename: str
    saved_path: str
    status: Literal["queued"]
    stats: Optional[Dict[str, int]] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to answer from indexed documents.")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="Maximum retrieved chunks.")
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional exact-match metadata filter applied during retrieval.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What methodology does the paper use?",
                    "top_k": 5,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str


class DependencyHealth(BaseModel):
    status: Literal["ok", "error"]
    message: Optional[str] = None


class ReadinessResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    checks: Dict[str, DependencyHealth]


class RuntimeConfigResponse(BaseModel):
    raw_data_dir: str
    processed_data_dir: str
    index_dir: str
    database_url: str
    redis_url: str
    task_queue_name: str
    embedding_provider: str
    embedding_model_name: str
    vector_store_provider: str
    use_faiss: bool
    top_k: int


DocumentStatus = Literal["uploaded", "processing", "indexed", "failed"]
TaskStatus = Literal["queued", "processing", "indexed", "failed"]


class DocumentResponse(BaseModel):
    id: str
    filename: str
    source_path: str
    content_type: Optional[str] = None
    status: DocumentStatus
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    indexed_at: Optional[datetime] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "58bfcf60-9ce1-4c17-9516-1336009a542b",
                    "filename": "paper.pdf",
                    "source_path": "data/raw/paper.pdf",
                    "content_type": "application/pdf",
                    "status": "indexed",
                    "error_message": None,
                    "created_at": "2026-04-30T12:00:00",
                    "updated_at": "2026-04-30T12:01:00",
                    "indexed_at": "2026-04-30T12:01:00",
                }
            ]
        }
    }


class IndexTaskResponse(BaseModel):
    id: str
    document_id: str
    status: TaskStatus
    stats: Optional[Dict[str, int]] = None
    error_message: Optional[str] = None
    retry_count: int
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "d62d1578-88c6-4af2-89a4-e92088bb6a90",
                    "document_id": "58bfcf60-9ce1-4c17-9516-1336009a542b",
                    "status": "indexed",
                    "stats": {"documents": 1, "chunks": 12},
                    "error_message": None,
                    "retry_count": 0,
                    "created_at": "2026-04-30T12:00:00",
                    "updated_at": "2026-04-30T12:01:00",
                    "completed_at": "2026-04-30T12:01:00",
                }
            ]
        }
    }


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]


class IndexTaskListResponse(BaseModel):
    tasks: List[IndexTaskResponse]


class RetryTaskResponse(BaseModel):
    task_id: str
    document_id: str
    status: Literal["queued"]
    retry_count: int


class ReindexDocumentResponse(BaseModel):
    document_id: str
    task_id: str
    status: Literal["queued"]


class DeleteDocumentResponse(BaseModel):
    document_id: str
    status: Literal["deleted"]
    stats: Dict[str, int]

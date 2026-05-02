from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api import repository
from src.api.database import get_db, init_db
from src.api.logging_config import configure_logging
from src.api.models import (
    DependencyHealth,
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    IndexTaskListResponse,
    IndexTaskResponse,
    QueryRequest,
    ReadinessResponse,
    ReindexDocumentResponse,
    RetryTaskResponse,
    RuntimeConfigResponse,
    UploadResponse,
)
from src.api.queue import get_redis
from src.api.service import EmptyUploadError, TaskEnqueueError, UnsupportedDocumentTypeError, rag_service
from src.config import get_settings
from src.schemas import AnswerResult

logger = logging.getLogger("citation_aware_rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    yield


app = FastAPI(
    title="Citation-Aware RAG API",
    version="0.1.0",
    description=(
        "Upload PDF/text documents, index them asynchronously with Redis/RQ, "
        "track document and task status, and query answers with chunk-level citations."
    ),
    lifespan=lifespan,
    contact={"name": "Citation-Aware RAG Maintainers"},
    license_info={"name": "Project local license"},
)


def _error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str | None = None,
    details: dict | None = None,
) -> JSONResponse:
    payload = ErrorResponse(error=ErrorDetail(code=code, message=message, request_id=request_id, details=details))
    response = JSONResponse(status_code=status_code, content=payload.model_dump())
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_id = request_id
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            status_code,
            duration_ms,
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict):
        code = str(detail.get("code", "http_error"))
        message = str(detail.get("message", "HTTP error."))
        details = detail.get("details")
    else:
        code = "http_error"
        message = str(detail)
        details = None
    return _error_response(
        status_code=exc.status_code,
        code=code,
        message=message,
        request_id=getattr(request.state, "request_id", None),
        details=details,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="validation_error",
        message="Request validation failed.",
        request_id=getattr(request.state, "request_id", None),
        details={"errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error")
    return _error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="internal_error",
        message="Internal server error.",
        request_id=getattr(request.state, "request_id", None),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Liveness check",
    description="Returns ok when the API process is running.",
)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="citation-aware-rag")


@app.get(
    "/health/ready",
    response_model=ReadinessResponse,
    tags=["system"],
    summary="Readiness check",
    description="Checks database and Redis connectivity used by the API and worker.",
)
def readiness(db: Session = Depends(get_db)) -> ReadinessResponse:
    checks: dict[str, DependencyHealth] = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = DependencyHealth(status="ok")
    except Exception as exc:
        checks["database"] = DependencyHealth(status="error", message=f"{type(exc).__name__}: {exc}")

    try:
        get_redis().ping()
        checks["redis"] = DependencyHealth(status="ok")
    except Exception as exc:
        checks["redis"] = DependencyHealth(status="error", message=f"{type(exc).__name__}: {exc}")

    overall = "ok" if all(item.status == "ok" for item in checks.values()) else "degraded"
    return ReadinessResponse(status=overall, service="citation-aware-rag", checks=checks)


@app.get(
    "/config",
    response_model=RuntimeConfigResponse,
    tags=["system"],
    summary="Runtime configuration",
    description="Returns non-secret runtime settings needed by API clients and operators.",
)
def runtime_config() -> RuntimeConfigResponse:
    settings = get_settings()
    return RuntimeConfigResponse(
        raw_data_dir=str(settings.raw_data_dir),
        processed_data_dir=str(settings.processed_data_dir),
        index_dir=str(settings.index_dir),
        database_url=settings.database_url,
        redis_url=settings.redis_url,
        task_queue_name=settings.task_queue_name,
        embedding_provider=settings.embedding_provider,
        embedding_model_name=settings.embedding_model_name,
        vector_store_provider=settings.vector_store_provider,
        use_faiss=settings.use_faiss,
        top_k=settings.top_k,
    )


@app.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["documents"],
    summary="Upload a document and enqueue indexing",
    description=(
        "Stores a PDF, Markdown, or text file, creates document/task status records, "
        "enqueues asynchronous indexing, and returns immediately with task_id."
    ),
)
def upload_document(request: Request, file: UploadFile, db: Session = Depends(get_db)) -> UploadResponse:
    try:
        document_id, task_id, saved_path = rag_service.upload_and_enqueue(db, file)
    except (EmptyUploadError, UnsupportedDocumentTypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_upload", "message": str(exc)},
        ) from exc
    except TaskEnqueueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "queue_unavailable", "message": str(exc)},
        ) from exc

    logger.info(
        "document_upload_enqueued request_id=%s document_id=%s task_id=%s filename=%s path=%s",
        getattr(request.state, "request_id", None),
        document_id,
        task_id,
        saved_path.name,
        saved_path,
    )
    return UploadResponse(
        document_id=document_id,
        task_id=task_id,
        filename=saved_path.name,
        saved_path=str(saved_path),
        status="queued",
    )


def _document_response(row) -> DocumentResponse:
    return DocumentResponse(
        id=row.id,
        filename=row.filename,
        source_path=row.source_path,
        content_type=row.content_type,
        status=row.status,
        error_message=row.error_message,
        created_at=row.created_at,
        updated_at=row.updated_at,
        indexed_at=row.indexed_at,
    )


def _task_response(row) -> IndexTaskResponse:
    return IndexTaskResponse(
        id=row.id,
        document_id=row.document_id,
        status=row.status,
        stats=row.stats,
        error_message=row.error_message,
        retry_count=row.retry_count,
        created_at=row.created_at,
        updated_at=row.updated_at,
        completed_at=row.completed_at,
    )


@app.get(
    "/documents",
    response_model=DocumentListResponse,
    tags=["documents"],
    summary="List uploaded documents",
    description="Returns persisted document metadata and indexing status, newest first.",
)
def list_documents(limit: int = 100, db: Session = Depends(get_db)) -> DocumentListResponse:
    return DocumentListResponse(documents=[_document_response(row) for row in repository.list_documents(db, limit)])


@app.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["documents"],
    summary="Get document status",
    description="Returns one uploaded document's metadata and processing status.",
)
def get_document(document_id: str, db: Session = Depends(get_db)) -> DocumentResponse:
    row = repository.get_document(db, document_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "document_not_found", "message": "Document not found."},
        )
    return _document_response(row)


@app.post(
    "/documents/{document_id}/reindex",
    response_model=ReindexDocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["documents"],
    summary="Reindex a document",
    description="Creates a new indexing task for an existing document. The worker replaces old chunks before indexing.",
)
def reindex_document(request: Request, document_id: str, db: Session = Depends(get_db)) -> ReindexDocumentResponse:
    try:
        reindexed_document_id, task_id = rag_service.reindex_document(db, document_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "document_not_found", "message": str(exc).strip("'")},
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "document_source_missing", "message": str(exc)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "document_not_reindexable", "message": str(exc)},
        ) from exc
    except TaskEnqueueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "queue_unavailable", "message": str(exc)},
        ) from exc

    logger.info(
        "document_reindex_enqueued request_id=%s document_id=%s task_id=%s",
        getattr(request.state, "request_id", None),
        reindexed_document_id,
        task_id,
    )
    return ReindexDocumentResponse(document_id=reindexed_document_id, task_id=task_id, status="queued")


@app.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
    tags=["documents"],
    summary="Delete a document",
    description="Deletes document metadata, source file, processed chunks, and vector rows for the document.",
)
def delete_document(request: Request, document_id: str, db: Session = Depends(get_db)) -> DeleteDocumentResponse:
    try:
        stats = rag_service.delete_document(db, document_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "document_not_found", "message": str(exc).strip("'")},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "document_not_deletable", "message": str(exc)},
        ) from exc

    logger.info(
        "document_delete_completed request_id=%s document_id=%s stats=%s",
        getattr(request.state, "request_id", None),
        document_id,
        stats,
    )
    return DeleteDocumentResponse(document_id=document_id, status="deleted", stats=stats)


@app.get(
    "/tasks",
    response_model=IndexTaskListResponse,
    tags=["tasks"],
    summary="List indexing tasks",
    description="Returns persisted indexing task status, newest first.",
)
def list_tasks(limit: int = 100, db: Session = Depends(get_db)) -> IndexTaskListResponse:
    return IndexTaskListResponse(tasks=[_task_response(row) for row in repository.list_index_tasks(db, limit)])


@app.get(
    "/tasks/{task_id}",
    response_model=IndexTaskResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["tasks"],
    summary="Get indexing task status",
    description="Returns queue/worker state, retry count, errors, and indexing stats for a task.",
)
def get_task(task_id: str, db: Session = Depends(get_db)) -> IndexTaskResponse:
    row = repository.get_index_task(db, task_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "task_not_found", "message": "Task not found."},
        )
    return _task_response(row)


@app.post(
    "/tasks/{task_id}/retry",
    response_model=RetryTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["tasks"],
    summary="Retry a failed indexing task",
    description="Re-enqueues a failed task and increments retry_count. Non-failed tasks return 409.",
)
def retry_task(request: Request, task_id: str, db: Session = Depends(get_db)) -> RetryTaskResponse:
    try:
        retried_task_id, document_id, retry_count = rag_service.retry_index_task(db, task_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "task_not_found", "message": str(exc).strip("'")},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "task_not_retryable", "message": str(exc)},
        ) from exc
    except TaskEnqueueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "queue_unavailable", "message": str(exc)},
        ) from exc

    logger.info(
        "task_retry_enqueued request_id=%s task_id=%s document_id=%s retry_count=%s",
        getattr(request.state, "request_id", None),
        retried_task_id,
        document_id,
        retry_count,
    )
    return RetryTaskResponse(
        task_id=retried_task_id,
        document_id=document_id,
        status="queued",
        retry_count=retry_count,
    )


@app.post(
    "/query",
    response_model=AnswerResult,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["query"],
    summary="Query indexed documents",
    description="Retrieves relevant chunks from the current index and returns a citation-aware answer.",
)
def query_documents(request: Request, payload: QueryRequest) -> AnswerResult:
    try:
        result = rag_service.answer(
            payload.query,
            top_k=payload.top_k,
            metadata_filter=payload.metadata_filter,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "index_not_found",
                "message": f"Index not found. Upload a document first or run scripts/build_index.py. {exc}",
            },
        ) from exc
    logger.info(
        "query_completed request_id=%s top_k=%s citations=%s generation_mode=%s",
        getattr(request.state, "request_id", None),
        payload.top_k,
        len(result.citations),
        result.generation_mode,
    )
    return result

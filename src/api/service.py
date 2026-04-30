from __future__ import annotations

import shutil
import threading
import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.orm import Session

from src.api import repository
from src.api.queue import enqueue_index_task
from src.config import Settings, get_settings
from src.pipeline import CitationAwareRAG
from src.schemas import AnswerResult
from src.utils import ensure_dir


SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".txt", ".md"}
logger = logging.getLogger("citation_aware_rag.service")


class UnsupportedDocumentTypeError(ValueError):
    pass


class EmptyUploadError(ValueError):
    pass


class TaskEnqueueError(RuntimeError):
    pass


class RAGService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._rag: Optional[CitationAwareRAG] = None
        self._lock = threading.RLock()

    def upload_and_enqueue(self, db: Session, upload: UploadFile) -> tuple[str, str, Path]:
        target_path = self._save_upload(upload)
        document = repository.create_document(
            db,
            filename=target_path.name,
            source_path=str(target_path),
            content_type=upload.content_type,
        )
        task = repository.create_index_task(db, document.id)
        db.commit()

        try:
            job_id = enqueue_index_task(task.id, settings=self.settings)
        except Exception as exc:
            db.rollback()
            error_message = f"{type(exc).__name__}: {exc}"
            repository.mark_index_failure(db, document, task, error_message)
            db.commit()
            raise TaskEnqueueError(error_message) from exc

        logger.info(
            "upload_saved_and_task_queued document_id=%s task_id=%s job_id=%s path=%s",
            document.id,
            task.id,
            job_id,
            target_path,
        )
        return document.id, task.id, target_path

    def retry_index_task(self, db: Session, task_id: str) -> tuple[str, str, int]:
        task = repository.get_index_task(db, task_id)
        if task is None:
            raise KeyError("Task not found.")
        document = repository.get_document(db, task.document_id)
        if document is None:
            raise KeyError("Document not found.")
        if task.status not in {"failed"}:
            raise ValueError("Only failed tasks can be retried.")

        repository.mark_task_retry_queued(db, document, task)
        db.commit()
        try:
            job_id = enqueue_index_task(task.id, settings=self.settings)
        except Exception as exc:
            db.rollback()
            error_message = f"{type(exc).__name__}: {exc}"
            repository.mark_index_failure(db, document, task, error_message)
            db.commit()
            raise TaskEnqueueError(error_message) from exc

        logger.info(
            "task_retry_queued document_id=%s task_id=%s retry_count=%s job_id=%s",
            document.id,
            task.id,
            task.retry_count,
            job_id,
        )
        return task.id, document.id, task.retry_count

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> AnswerResult:
        with self._lock:
            return self._get_rag().answer(query, top_k=top_k, metadata_filter=metadata_filter)

    def _get_rag(self) -> CitationAwareRAG:
        if self._rag is None:
            self._rag = CitationAwareRAG(self.settings)
        return self._rag

    def _save_upload(self, upload: UploadFile) -> Path:
        original_name = Path(upload.filename or "").name
        if not original_name:
            raise EmptyUploadError("Uploaded file must have a filename.")

        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
            allowed = ", ".join(sorted(SUPPORTED_UPLOAD_SUFFIXES))
            raise UnsupportedDocumentTypeError(f"Unsupported file type '{suffix}'. Allowed types: {allowed}.")

        ensure_dir(self.settings.raw_data_dir)
        target_path = self.settings.raw_data_dir / original_name
        if target_path.exists():
            target_path = self.settings.raw_data_dir / f"{target_path.stem}-{uuid4().hex[:8]}{target_path.suffix}"

        with target_path.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)

        if target_path.stat().st_size == 0:
            target_path.unlink(missing_ok=True)
            raise EmptyUploadError("Uploaded file is empty.")

        return target_path


rag_service = RAGService()

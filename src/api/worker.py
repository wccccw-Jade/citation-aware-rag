from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy.orm import Session

from src.api import repository
from src.api.database import SessionLocal, init_db
from src.config import Settings, get_settings
from src.pipeline import CitationAwareRAG

logger = logging.getLogger("citation_aware_rag.worker")


@contextmanager
def _session_scope() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def process_index_task(task_id: str) -> dict[str, int]:
    init_db()
    settings = get_settings()
    logger.info("index_task_started task_id=%s", task_id)

    with _session_scope() as db:
        task = repository.get_index_task(db, task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")
        document = repository.get_document(db, task.document_id)
        if document is None:
            raise ValueError(f"Document not found for task: {task_id}")
        repository.mark_index_processing(db, document, task)

    try:
        stats = _index_document(settings, Path(document.source_path))
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        with _session_scope() as db:
            task = repository.get_index_task(db, task_id)
            if task is not None:
                document = repository.get_document(db, task.document_id)
                if document is not None:
                    repository.mark_index_failure(db, document, task, error_message)
        logger.exception("Index task failed task_id=%s", task_id)
        raise

    with _session_scope() as db:
        task = repository.get_index_task(db, task_id)
        if task is None:
            raise ValueError(f"Task disappeared during processing: {task_id}")
        document = repository.get_document(db, task.document_id)
        if document is None:
            raise ValueError(f"Document disappeared during processing: {task_id}")
        repository.mark_index_success(db, document, task, stats)

    logger.info("index_task_completed task_id=%s stats=%s", task_id, stats)
    return stats


def _build_index(settings: Settings) -> dict[str, int]:
    rag = CitationAwareRAG(settings)
    return rag.build_index()


def _index_document(settings: Settings, document_path: Path) -> dict[str, int]:
    rag = CitationAwareRAG(settings)
    return rag.reindex_document(document_path)

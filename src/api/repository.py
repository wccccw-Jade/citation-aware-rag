from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from sqlalchemy import desc
from sqlalchemy.orm import Session

from src.api.db_models import DocumentORM, IndexTaskORM


def create_document(
    db: Session,
    filename: str,
    source_path: str,
    content_type: Optional[str],
) -> DocumentORM:
    document = DocumentORM(
        id=str(uuid4()),
        filename=filename,
        source_path=source_path,
        content_type=content_type,
        status="uploaded",
    )
    db.add(document)
    db.flush()
    return document


def create_index_task(db: Session, document_id: str) -> IndexTaskORM:
    task = IndexTaskORM(id=str(uuid4()), document_id=document_id, status="queued")
    db.add(task)
    db.flush()
    return task


def mark_index_queued(db: Session, document: DocumentORM, task: IndexTaskORM) -> None:
    document.status = "uploaded"
    document.error_message = None
    task.status = "queued"
    task.error_message = None
    task.completed_at = None
    db.flush()


def mark_document_processing(db: Session, document: DocumentORM) -> None:
    document.status = "processing"
    document.error_message = None
    db.flush()


def mark_index_processing(db: Session, document: DocumentORM, task: IndexTaskORM) -> None:
    document.status = "processing"
    document.error_message = None
    task.status = "processing"
    task.error_message = None
    task.completed_at = None
    db.flush()


def mark_index_success(db: Session, document: DocumentORM, task: IndexTaskORM, stats: Dict[str, int]) -> None:
    now = datetime.utcnow()
    document.status = "indexed"
    document.error_message = None
    document.indexed_at = now
    task.status = "indexed"
    task.stats = stats
    task.error_message = None
    task.completed_at = now
    db.flush()


def mark_index_failure(db: Session, document: DocumentORM, task: IndexTaskORM, error_message: str) -> None:
    now = datetime.utcnow()
    document.status = "failed"
    document.error_message = error_message
    task.status = "failed"
    task.error_message = error_message
    task.completed_at = now
    db.flush()


def mark_task_retry_queued(db: Session, document: DocumentORM, task: IndexTaskORM) -> None:
    document.status = "uploaded"
    document.error_message = None
    task.status = "queued"
    task.error_message = None
    task.stats = None
    task.retry_count = (task.retry_count or 0) + 1
    task.completed_at = None
    db.flush()


def mark_document_reindex_queued(db: Session, document: DocumentORM, task: IndexTaskORM) -> None:
    document.status = "uploaded"
    document.error_message = None
    document.indexed_at = None
    task.status = "queued"
    task.error_message = None
    task.stats = None
    task.completed_at = None
    db.flush()


def list_documents(db: Session, limit: int = 100) -> List[DocumentORM]:
    return db.query(DocumentORM).order_by(desc(DocumentORM.created_at)).limit(limit).all()


def get_document(db: Session, document_id: str) -> Optional[DocumentORM]:
    return db.query(DocumentORM).filter(DocumentORM.id == document_id).one_or_none()


def list_index_tasks(db: Session, limit: int = 100) -> List[IndexTaskORM]:
    return db.query(IndexTaskORM).order_by(desc(IndexTaskORM.created_at)).limit(limit).all()


def get_index_task(db: Session, task_id: str) -> Optional[IndexTaskORM]:
    return db.query(IndexTaskORM).filter(IndexTaskORM.id == task_id).one_or_none()


def delete_document(db: Session, document: DocumentORM) -> None:
    db.query(IndexTaskORM).filter(IndexTaskORM.document_id == document.id).delete(synchronize_session=False)
    db.delete(document)
    db.flush()

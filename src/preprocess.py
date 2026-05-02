from __future__ import annotations

import json
from pathlib import Path

from .pdf_parser import parse_document
from .schemas import DocumentRecord
from .utils import ensure_dir, read_jsonl, write_jsonl


def load_documents(raw_dir: Path) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    for path in sorted(raw_dir.glob("*")):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}:
            documents.extend(load_document(path))
    return documents


def load_document(path: Path) -> list[DocumentRecord]:
    return parse_document(path)


def persist_documents(documents: list[DocumentRecord], output_dir: Path) -> None:
    ensure_dir(output_dir)
    write_jsonl(output_dir / "documents.jsonl", [doc.model_dump() for doc in documents])
    metadata = {
        "document_count": len(documents),
        "source_files": sorted({doc.source_path for doc in documents}),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def merge_persisted_documents(documents: list[DocumentRecord], output_dir: Path) -> int:
    ensure_dir(output_dir)
    path = output_dir / "documents.jsonl"
    existing = read_jsonl(path) if path.exists() else []
    existing_doc_ids = {row["doc_id"] for row in existing}
    new_rows = [doc.model_dump() for doc in documents if doc.doc_id not in existing_doc_ids]
    if new_rows:
        write_jsonl(path, [*existing, *new_rows])
    elif not path.exists():
        write_jsonl(path, [])

    merged = [*existing, *new_rows]
    metadata = {
        "document_count": len(merged),
        "source_files": sorted({row["source_path"] for row in merged}),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return len(new_rows)


def remove_persisted_documents_by_source(source_path: str, output_dir: Path) -> dict[str, int]:
    ensure_dir(output_dir)
    path = output_dir / "documents.jsonl"
    existing = read_jsonl(path) if path.exists() else []
    remaining = [row for row in existing if row["source_path"] != source_path]
    removed_count = len(existing) - len(remaining)
    write_jsonl(path, remaining)
    metadata = {
        "document_count": len(remaining),
        "source_files": sorted({row["source_path"] for row in remaining}),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"documents": removed_count}

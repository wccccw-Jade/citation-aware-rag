from __future__ import annotations

import json
from pathlib import Path

from .pdf_parser import parse_document
from .schemas import DocumentRecord
from .utils import ensure_dir, write_jsonl


def load_documents(raw_dir: Path) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    for path in sorted(raw_dir.glob("*")):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}:
            documents.extend(parse_document(path))
    return documents


def persist_documents(documents: list[DocumentRecord], output_dir: Path) -> None:
    ensure_dir(output_dir)
    write_jsonl(output_dir / "documents.jsonl", [doc.model_dump() for doc in documents])
    metadata = {
        "document_count": len(documents),
        "source_files": sorted({doc.source_path for doc in documents}),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

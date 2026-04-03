from __future__ import annotations

from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    from PyPDF2 import PdfReader

from .schemas import DocumentRecord
from .utils import stable_id


def parse_document(path: Path) -> list[DocumentRecord]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix in {".txt", ".md"}:
        return _parse_text(path)
    raise ValueError(f"Unsupported file type: {path}")


def _parse_pdf(path: Path) -> list[DocumentRecord]:
    reader = PdfReader(str(path))
    title = path.stem
    records: list[DocumentRecord] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        doc_id = stable_id(str(path), str(index))
        records.append(
            DocumentRecord(
                doc_id=doc_id,
                source_path=str(path),
                title=title,
                page_number=index,
                text=text,
                metadata={"file_type": "pdf"},
            )
        )
    return records


def _parse_text(path: Path) -> list[DocumentRecord]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [
        DocumentRecord(
            doc_id=stable_id(str(path), "1"),
            source_path=str(path),
            title=path.stem,
            page_number=1,
            text=text,
            metadata={"file_type": path.suffix.lower().lstrip(".")},
        )
    ]

from __future__ import annotations

import re

from .schemas import ChunkRecord, DocumentRecord
from .utils import stable_id

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n+")


def _split_into_units(text: str) -> list[str]:
    paragraphs = [part.strip() for part in PARAGRAPH_SPLIT_PATTERN.split(text) if part.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    return [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]


def chunk_documents(
    documents: list[DocumentRecord],
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for doc in documents:
        text = doc.text.strip()
        if not text:
            continue

        units = _split_into_units(text)
        if not units:
            continue

        current_units: list[str] = []
        current_length = 0
        chunk_start = 0

        for unit in units:
            separator = 1 if current_units else 0
            projected_length = current_length + separator + len(unit)
            if current_units and projected_length > chunk_size:
                chunk_text = " ".join(current_units).strip()
                chunk_end = chunk_start + len(chunk_text)
                chunk_id = stable_id(doc.doc_id, str(chunk_start), str(chunk_end))
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        source_path=doc.source_path,
                        title=doc.title,
                        text=chunk_text,
                        page_number=doc.page_number,
                        start_char=chunk_start,
                        end_char=chunk_end,
                        metadata=dict(doc.metadata),
                    )
                )

                overlap_units: list[str] = []
                overlap_length = 0
                for previous in reversed(current_units):
                    extra = len(previous) + (1 if overlap_units else 0)
                    if overlap_units and overlap_length + extra > chunk_overlap:
                        break
                    overlap_units.insert(0, previous)
                    overlap_length += extra
                current_units = overlap_units
                current_length = sum(len(item) for item in current_units) + max(0, len(current_units) - 1)
                chunk_start = max(0, chunk_end - current_length)

            if len(unit) > chunk_size and not current_units:
                for start in range(0, len(unit), chunk_size):
                    piece = unit[start : start + chunk_size].strip()
                    if not piece:
                        continue
                    piece_start = chunk_start + start
                    piece_end = piece_start + len(piece)
                    chunk_id = stable_id(doc.doc_id, str(piece_start), str(piece_end))
                    chunks.append(
                        ChunkRecord(
                            chunk_id=chunk_id,
                            doc_id=doc.doc_id,
                            source_path=doc.source_path,
                            title=doc.title,
                            text=piece,
                            page_number=doc.page_number,
                            start_char=piece_start,
                            end_char=piece_end,
                            metadata=dict(doc.metadata),
                        )
                    )
                chunk_start += len(unit)
                current_units = []
                current_length = 0
                continue

            if current_units:
                current_length += 1
            current_units.append(unit)
            current_length += len(unit)

        if current_units:
            chunk_text = " ".join(current_units).strip()
            chunk_end = chunk_start + len(chunk_text)
            chunk_id = stable_id(doc.doc_id, str(chunk_start), str(chunk_end))
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source_path=doc.source_path,
                    title=doc.title,
                    text=chunk_text,
                    page_number=doc.page_number,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    metadata=dict(doc.metadata),
                )
            )
    return chunks

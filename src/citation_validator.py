from __future__ import annotations

import re
from typing import Any

from .schemas import RetrievedChunk

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
QUOTE_PATTERN = re.compile(r'"([^"]{12,})"\s*\[(\d+)\]')
REFUSAL_PATTERNS = (
    "i don't know based on the provided documents",
    "no grounded evidence was found",
    "no grounded answer could be composed",
    "not enough evidence",
)


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _split_claims(answer: str) -> list[str]:
    claims: list[str] = []
    for line in answer.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("grounded answer:"):
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        claims.extend(part.strip("- ").strip() for part in parts if part.strip("- ").strip())
    return claims


def _is_refusal(answer: str) -> bool:
    normalized = _normalize_text(answer)
    return any(pattern in normalized for pattern in REFUSAL_PATTERNS)


def _quote_in_chunk(quote: str, chunk_text: str) -> bool:
    return _normalize_text(quote) in _normalize_text(chunk_text)


def validate_citations(answer: str, retrieved_chunks: list[RetrievedChunk]) -> dict[str, Any]:
    valid_labels = set(range(1, len(retrieved_chunks) + 1))
    chunk_by_label = {label: item.chunk for label, item in enumerate(retrieved_chunks, start=1)}
    cited_labels = [int(match) for match in CITATION_PATTERN.findall(answer)]
    unique_cited_labels = sorted(set(cited_labels))
    invalid_labels = [label for label in unique_cited_labels if label not in valid_labels]
    has_citation = bool(cited_labels)
    refusal = _is_refusal(answer)

    quote_checks: list[dict[str, Any]] = []
    for quote, label_text in QUOTE_PATTERN.findall(answer):
        label = int(label_text)
        chunk = chunk_by_label.get(label)
        found = bool(chunk and _quote_in_chunk(quote, chunk.text))
        quote_checks.append({"label": label, "quote": quote, "found": found})

    unsupported_claims: list[str] = []
    if not refusal:
        for claim in _split_claims(answer):
            if not CITATION_PATTERN.search(claim):
                unsupported_claims.append(claim)

    valid_quote_checks = all(check["found"] for check in quote_checks)
    valid = (
        (refusal or has_citation)
        and not invalid_labels
        and valid_quote_checks
        and not unsupported_claims
    )

    reasons: list[str] = []
    if not has_citation and not refusal:
        reasons.append("answer_has_no_citations")
    if invalid_labels:
        reasons.append("answer_references_missing_labels")
    if not valid_quote_checks:
        reasons.append("supporting_quote_not_found")
    if unsupported_claims:
        reasons.append("unsupported_claim_without_citation")

    return {
        "valid": valid,
        "has_citation": has_citation,
        "refusal": refusal,
        "cited_labels": unique_cited_labels,
        "invalid_labels": invalid_labels,
        "invalid_citation_count": len(invalid_labels),
        "quote_checks": quote_checks,
        "unsupported_claims": unsupported_claims,
        "unsupported_claim_count": len(unsupported_claims),
        "reasons": reasons,
    }


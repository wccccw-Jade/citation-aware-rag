from src.citation_validator import validate_citations
from tests.test_generator import _retrieved_chunk


def test_validate_citations_accepts_valid_label() -> None:
    result = validate_citations("RAG retrieves evidence before generation [1].", [_retrieved_chunk()])

    assert result["valid"] is True
    assert result["has_citation"] is True
    assert result["invalid_labels"] == []


def test_validate_citations_rejects_missing_label() -> None:
    result = validate_citations("RAG retrieves evidence before generation [9].", [_retrieved_chunk()])

    assert result["valid"] is False
    assert result["invalid_labels"] == [9]
    assert "answer_references_missing_labels" in result["reasons"]


def test_validate_citations_rejects_quote_not_found() -> None:
    result = validate_citations('"This exact quote is absent from the chunk" [1].', [_retrieved_chunk()])

    assert result["valid"] is False
    assert result["quote_checks"][0]["found"] is False
    assert "supporting_quote_not_found" in result["reasons"]


def test_validate_citations_allows_refusal_without_citation() -> None:
    result = validate_citations("I don't know based on the provided documents.", [])

    assert result["valid"] is True
    assert result["refusal"] is True
    assert result["has_citation"] is False


def test_validate_citations_flags_claim_without_citation() -> None:
    result = validate_citations("RAG retrieves evidence. This second claim has no citation.", [_retrieved_chunk()])

    assert result["valid"] is False
    assert "answer_has_no_citations" in result["reasons"]
    assert result["unsupported_claim_count"] == 2


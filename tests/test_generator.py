from pathlib import Path

from src.config import Settings
from src.generator import generate_answer
from src.schemas import ChunkRecord, RetrievedChunk


class MockLLMProvider:
    provider_name = "mock"
    model_name = "mock-model"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        assert "Answer only using the retrieved context" in system_prompt
        assert "Retrieved context:" in user_prompt
        return "RAG retrieves evidence before generation [1]."


class FailingLLMProvider:
    provider_name = "mock"
    model_name = "mock-model"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("network unavailable")


def _retrieved_chunk() -> RetrievedChunk:
    text = "RAG retrieves relevant evidence before generating an answer."
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_path="data/raw/1.pdf",
        title=Path("data/raw/1.pdf").stem,
        text=text,
        page_number=3,
        start_char=0,
        end_char=len(text),
        metadata={"file_type": "pdf"},
    )
    return RetrievedChunk(chunk=chunk, score=0.9)


def test_generate_answer_falls_back_without_llm_key() -> None:
    settings = Settings(use_llm_generation=True, openai_api_key="")

    result = generate_answer("What does RAG retrieve?", [_retrieved_chunk()], settings=settings)

    assert result.answer.startswith("Grounded answer:")
    assert "[1]" in result.answer
    assert result.citations[0]["generation_mode"] == "extractive"


def test_generate_answer_uses_mock_llm_provider() -> None:
    result = generate_answer("What does RAG retrieve?", [_retrieved_chunk()], llm_provider=MockLLMProvider())

    assert result.answer == "RAG retrieves evidence before generation [1]."
    assert result.citations[0]["generation_mode"] == "mock"


def test_llm_answer_citations_still_come_from_retrieved_chunks() -> None:
    retrieved = _retrieved_chunk()

    result = generate_answer("What does RAG retrieve?", [retrieved], llm_provider=MockLLMProvider())

    citation = result.citations[0]
    assert citation["label"] == 1
    assert citation["chunk_id"] == retrieved.chunk.chunk_id
    assert citation["source_path"] == retrieved.chunk.source_path
    assert citation["page_number"] == retrieved.chunk.page_number


def test_generate_answer_falls_back_when_llm_call_fails() -> None:
    result = generate_answer("What does RAG retrieve?", [_retrieved_chunk()], llm_provider=FailingLLMProvider())

    assert result.answer.startswith("Grounded answer:")
    assert result.citations[0]["generation_mode"] == "extractive_fallback_after_llm_error"
    assert "RuntimeError" in result.citations[0]["llm_error"]

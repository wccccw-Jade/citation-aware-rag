from pathlib import Path

from src.config import Settings
from src.embedding import LocalEmbeddingModel
from src.retrieval import Retriever
from src.schemas import ChunkRecord
from src.vector_store import VectorStore


def _chunk(index: int, text: str, source_path: str = "a.pdf") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"chunk-{index}",
        doc_id=f"doc-{index}",
        source_path=source_path,
        title=Path(source_path).stem,
        text=text,
        page_number=index,
        start_char=0,
        end_char=len(text),
        metadata={"file_type": "pdf"},
    )


def test_retriever_combines_dense_and_lexical_results(tmp_path) -> None:
    settings = Settings(retrieval_candidate_pool=3, top_k=2)
    embedder = LocalEmbeddingModel(dim=64)
    chunks = [
        _chunk(1, "RAG systems retrieve grounded evidence for generation."),
        _chunk(2, "Fine tuning stores knowledge in model weights."),
        _chunk(3, "Citation tracking maps answers to source pages."),
    ]
    store = VectorStore(tmp_path, use_faiss=False)
    store.build(embedder.encode([f"{chunk.title} {Path(chunk.source_path).name} {chunk.text}" for chunk in chunks]), [c.model_dump() for c in chunks])

    retriever = Retriever(embedder, store, settings)
    results = retriever.retrieve("What retrieves grounded evidence?", chunks, top_k=2)

    assert results
    assert results[0].chunk.chunk_id == "chunk-1"


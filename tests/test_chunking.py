from src.chunking import chunk_documents
from src.schemas import DocumentRecord


def test_chunk_documents_preserves_page_and_source_metadata() -> None:
    document = DocumentRecord(
        doc_id="doc-1",
        source_path="data/ragpapers/raw/example.pdf",
        title="example",
        page_number=7,
        text="First sentence. Second sentence. Third sentence.",
        metadata={"file_type": "pdf"},
    )

    chunks = chunk_documents([document], chunk_size=30, chunk_overlap=10)

    assert len(chunks) >= 2
    assert chunks[0].page_number == 7
    assert chunks[0].source_path == "data/ragpapers/raw/example.pdf"
    assert chunks[0].metadata["file_type"] == "pdf"

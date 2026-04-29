import numpy as np

from src.vector_store import VectorStore


def test_vector_store_numpy_search_supports_metadata_filter(tmp_path) -> None:
    store = VectorStore(tmp_path, use_faiss=False)
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    metadata = [
        {"source_path": "a.pdf", "metadata": {"kind": "paper"}},
        {"source_path": "b.pdf", "metadata": {"kind": "note"}},
        {"source_path": "c.pdf", "metadata": {"kind": "paper"}},
    ]

    store.build(embeddings, metadata, index_config={"embedding_model_name": "test"})
    store.load()

    results = store.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=2, metadata_filter={"kind": "paper"})

    assert [index for index, _ in results] == [0, 2]
    assert store.index_config["embedding_model_name"] == "test"


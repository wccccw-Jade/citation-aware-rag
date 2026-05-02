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


def test_vector_store_append_adds_only_new_chunk_ids(tmp_path) -> None:
    store = VectorStore(tmp_path, use_faiss=False)
    store.build(
        np.array([[1.0, 0.0]], dtype=np.float32),
        [{"chunk_id": "a", "source_path": "a.txt"}],
        index_config={"embedding_provider": "local", "embedding_model_name": "hashing", "embedding_dim": 2},
    )

    added = store.append(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [{"chunk_id": "a", "source_path": "a.txt"}, {"chunk_id": "b", "source_path": "b.txt"}],
        index_config={"embedding_provider": "local", "embedding_model_name": "hashing", "embedding_dim": 2},
    )

    store.load()

    assert added == 1
    assert [row["chunk_id"] for row in store.metadata] == ["a", "b"]
    assert store.embeddings.shape == (2, 2)


def test_vector_store_delete_by_source_path_removes_metadata_and_embeddings(tmp_path) -> None:
    store = VectorStore(tmp_path, use_faiss=False)
    store.build(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [{"chunk_id": "a", "source_path": "a.txt"}, {"chunk_id": "b", "source_path": "b.txt"}],
        index_config={"embedding_provider": "local", "embedding_model_name": "hashing", "embedding_dim": 2},
    )

    removed = store.delete_by_source_path("a.txt")
    store.load()

    assert removed == 1
    assert [row["chunk_id"] for row in store.metadata] == ["b"]
    assert store.embeddings.shape == (1, 2)

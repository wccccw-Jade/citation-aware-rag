import numpy as np

from src.config import Settings
from src.embedding import LocalEmbeddingModel, create_embedding_model


def test_local_embedding_is_deterministic_and_normalized() -> None:
    model = LocalEmbeddingModel(dim=32)

    first = model.encode(["retrieval augmented generation"])
    second = model.encode(["retrieval augmented generation"])

    assert first.shape == (1, 32)
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first[0]), 1.0)


def test_embedding_factory_returns_local_provider() -> None:
    settings = Settings(embedding_provider="local", embedding_dim=16)

    model = create_embedding_model(settings)

    assert model.provider_name == "local"
    assert model.dim == 16


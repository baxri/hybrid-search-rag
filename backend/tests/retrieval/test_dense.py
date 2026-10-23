import numpy as np
from pytest import approx

from app.retrieval.dense import DenseRetriever
from tests.fixtures import FIXTURE_DOCS, build_fixture_data


async def test_dense_returns_cosine_top_k(tmp_path):
    data_dir = build_fixture_data(tmp_path)
    d3_vector = np.load(data_dir / "embeddings.npy")[3]

    async def embed(query: str) -> np.ndarray:
        return d3_vector

    retriever = DenseRetriever(data_dir / "embeddings.npy", list(FIXTURE_DOCS), embed)
    results = await retriever.search("anything", k=4)

    assert results[0][0] == "d3"
    assert results[0][1] == approx(1.0, abs=1e-5)
    assert len(results) == 4

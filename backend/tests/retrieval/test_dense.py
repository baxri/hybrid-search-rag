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


def test_loading_does_not_hold_a_second_copy_of_the_embeddings(tmp_path):
    # Production loads 354 MB of vectors on a 1 GB box; a normalized copy would double the peak.
    import tracemalloc

    vectors = np.random.default_rng(0).random((4000, 64), dtype=np.float32)  # ~1 MB
    path = tmp_path / "embeddings.npy"
    np.save(path, vectors)

    tracemalloc.start()
    retriever = DenseRetriever(path, [str(i) for i in range(len(vectors))], embed=None)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 1.5 * vectors.nbytes
    assert np.allclose(np.linalg.norm(retriever._embeddings, axis=1), 1.0, atol=1e-5)

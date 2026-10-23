from app.retrieval.bm25 import BM25Retriever
from tests.fixtures import build_fixture_data


def test_bm25_finds_exact_term(tmp_path):
    retriever = BM25Retriever(build_fixture_data(tmp_path) / "bm25")

    results = retriever.search("1099-K form", k=3)

    assert results[0][0] == "d1"
    assert len(results) == 3

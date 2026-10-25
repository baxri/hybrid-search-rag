import json

import numpy as np
import pandas as pd
import pytest

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.corpus import Corpus
from app.retrieval.dense import DenseRetriever
from scripts.build_indexes import build_indexes, chunk_text, load_documents, to_chunks

DIM = 4


async def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic vectors from the text, so a query equal to a chunk finds that chunk."""
    rng = [np.random.default_rng(abs(hash(t)) % 2**32) for t in texts]
    return np.array([r.normal(size=DIM) for r in rng], dtype=np.float32)


def test_short_text_is_one_chunk():
    assert chunk_text("  Keep cash in a savings account.  ") == ["Keep cash in a savings account."]


def test_long_text_is_split_into_overlapping_windows():
    words = [f"w{i}" for i in range(10)]

    chunks = chunk_text(" ".join(words), chunk_words=4, overlap_words=1)

    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_words=3, overlap_words=3)


def test_chunk_ids_keep_short_docs_and_number_long_ones():
    docs = pd.DataFrame({"id": ["a", "b"], "title": ["A", ""], "text": ["one two", "x y z q"]})

    corpus = to_chunks(docs, chunk_words=3, overlap_words=1)

    assert corpus["_id"].tolist() == ["a", "b#1", "b#2"]
    assert corpus["title"].tolist() == ["A", "", ""]
    assert corpus["text"].tolist() == ["one two", "x y z", "z q"]


@pytest.mark.parametrize("suffix", [".csv", ".jsonl", ".parquet"])
def test_loads_each_format_with_custom_columns(tmp_path, suffix):
    frame = pd.DataFrame({"key": [1, 2], "body": ["first", None]})
    path = tmp_path / f"docs{suffix}"
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix == ".jsonl":
        path.write_text("\n".join(json.dumps(r) for r in frame.to_dict("records")))
    else:
        frame.to_parquet(path)

    docs = load_documents(path, id_column="key", text_column="body")

    assert docs.to_dict("list") == {"id": ["1", "2"], "title": ["", ""], "text": ["first", ""]}


def test_rejects_missing_columns_and_repeated_ids(tmp_path):
    path = tmp_path / "docs.csv"
    pd.DataFrame({"id": ["a"], "body": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no column"):
        load_documents(path)

    pd.DataFrame({"id": ["a", "a"], "text": ["x", "y"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_documents(path)


async def test_built_indexes_load_and_search_in_the_app(tmp_path):
    corpus = pd.DataFrame(
        {
            "_id": ["etf", "roth", "blank"],
            "title": ["", "", ""],
            "text": ["ETFs disclose their holdings daily.", "Convert to a Roth IRA.", " "],
        }
    )

    await build_indexes(corpus, tmp_path, fake_embed)

    loaded = Corpus.load(tmp_path)
    assert loaded.ids == ["etf", "roth", "blank"]
    assert BM25Retriever(tmp_path / "bm25").search("roth conversion", 1)[0][0] == "roth"

    async def embed(query: str) -> np.ndarray:
        return (await fake_embed([query]))[0]

    dense = DenseRetriever(tmp_path / "embeddings.npy", loaded.ids, embed)
    assert (await dense.search("ETFs disclose their holdings daily.", 1))[0][0] == "etf"
    assert np.load(tmp_path / "embeddings.npy").shape == (3, DIM)

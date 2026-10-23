"""A tiny on-disk corpus with the same layout as the real FiQA data dir."""

from pathlib import Path

import bm25s
import numpy as np
import pandas as pd

FIXTURE_DOCS = {
    "d0": "Keep your emergency fund in a high-yield savings account you can reach within a day.",
    "d1": "Payment apps send a 1099-K form once you pass the reporting threshold.",
    "d2": "Index funds have low fees and broad diversification.",
    "d3": "Short-term CDs can hold part of an emergency fund for a bit more yield.",
    "d4": "   ",
    "d5": "Paying off a mortgage early is a guaranteed return equal to its rate.",
}
EMBEDDING_DIM = 8


def build_fixture_data(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    ids = list(FIXTURE_DOCS)
    texts = list(FIXTURE_DOCS.values())

    pd.DataFrame({"_id": ids, "title": [""] * len(ids), "text": texts}).to_parquet(
        data_dir / "corpus.parquet"
    )

    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(texts, stopwords="en"))
    bm25_dir = data_dir / "bm25"
    retriever.save(str(bm25_dir))
    (bm25_dir / "doc_ids.txt").write_text("\n".join(ids))

    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(len(ids), EMBEDDING_DIM)).astype(np.float32)
    np.save(data_dir / "embeddings.npy", embeddings)
    return data_dir

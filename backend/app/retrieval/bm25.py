"""Sparse retrieval: BM25 keyword scoring. Wins on exact terms (tickers, form codes)."""

from pathlib import Path

import bm25s


class BM25Retriever:
    def __init__(self, index_dir: Path) -> None:
        self._retriever = bm25s.BM25.load(str(index_dir))
        self._doc_ids = (index_dir / "doc_ids.txt").read_text().splitlines()

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        indices, scores = self._retriever.retrieve(tokens, k=k, show_progress=False)
        return [(self._doc_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0].tolist())]

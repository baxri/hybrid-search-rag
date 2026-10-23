"""The FiQA corpus: forum posts keyed by doc id, in the row order the indexes were built from."""

from pathlib import Path

import pandas as pd


class Corpus:
    def __init__(self, ids: list[str], texts: list[str]) -> None:
        self.ids = ids
        self._text_by_id = dict(zip(ids, texts, strict=True))

    @classmethod
    def load(cls, data_dir: Path) -> "Corpus":
        frame = pd.read_parquet(data_dir / "corpus.parquet", columns=["_id", "text"])
        return cls(frame["_id"].tolist(), frame["text"].fillna("").tolist())

    def text(self, doc_id: str) -> str:
        return self._text_by_id[doc_id]

    def is_blank(self, doc_id: str) -> bool:
        # ~38 FiQA posts are empty; they were embedded as "[empty document]".
        return not self._text_by_id[doc_id].strip()

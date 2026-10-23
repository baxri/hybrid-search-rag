"""Citation numbering for one assistant turn.

Numbers are assigned in code, not by the model: every post a search returns gets a stable
[n] for the rest of the turn, and the frontend only renders markers that match a Source.
"""

import re

from app.retrieval.corpus import Corpus
from app.schemas import SearchTrace, Source

MAX_SOURCE_CHARS_FOR_MODEL = 2000

# [1], [1,2], [1, 2] -- plus any space before the group, so "CDs [1]." becomes "CDs."
_MARKER_GROUP = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


class SourceRegistry:
    """Hands out citation numbers within one turn; a post keeps its first number."""

    def __init__(self) -> None:
        self._by_doc_id: dict[str, Source] = {}

    @property
    def all(self) -> list[Source]:
        return list(self._by_doc_id.values())

    def add(self, trace: SearchTrace, corpus: Corpus) -> tuple[list[Source], list[Source]]:
        """Register the reranked posts of one search. Returns (this_search, newly_numbered)."""
        rerank = next(stage for stage in trace.stages if stage.name == "rerank")
        this_search: list[Source] = []
        newly_numbered: list[Source] = []
        for doc in rerank.results:
            source = self._by_doc_id.get(doc.doc_id)
            if source is None:
                source = Source(
                    n=len(self._by_doc_id) + 1, doc_id=doc.doc_id, text=corpus.text(doc.doc_id)
                )
                self._by_doc_id[doc.doc_id] = source
                newly_numbered.append(source)
            this_search.append(source)
        return this_search, newly_numbered


def format_for_model(sources: list[Source]) -> str:
    if not sources:
        return "No relevant posts found."
    return "\n\n".join(f"[{s.n}] {s.text[:MAX_SOURCE_CHARS_FOR_MODEL]}" for s in sources)


def strip_markers(text: str) -> str:
    return _MARKER_GROUP.sub("", text)

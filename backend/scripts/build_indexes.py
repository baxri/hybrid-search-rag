"""Build the search indexes from your own documents: corpus, BM25 index and embeddings.

Input is a .csv, .jsonl or .parquet file with an id and a text column (a title is optional).
Long documents are split into overlapping chunks of words, and each chunk is searched and
cited on its own as "<id>#<n>"; a document that fits in one chunk keeps its id.

    uv run python -m scripts.build_indexes --input docs.jsonl --out data
        [--id-column id] [--text-column text] [--title-column title]
        [--chunk-words 300] [--overlap-words 50]

Writes the layout the app loads: <out>/corpus.parquet, <out>/bm25/, <out>/embeddings.npy.
Embedding calls OpenAI with OPENAI_API_KEY from backend/.env (about $0.03 per million words).
"""

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
from openai import AsyncOpenAI, RateLimitError

from app.config import get_settings
from app.retrieval.dense import EMBEDDING_MODEL

EMBED_BATCH = 256
EMBED_CONCURRENCY = 2
RATE_LIMIT_WAIT_S = 20.0

EmbedBatch = Callable[[list[str]], Awaitable[np.ndarray]]


def load_documents(
    path: Path, id_column: str = "id", text_column: str = "text", title_column: str = "title"
) -> pd.DataFrame:
    """Read the input file into `id, title, text` columns (title is "" when missing)."""
    readers = {".csv": pd.read_csv, ".jsonl": lambda p: pd.read_json(p, lines=True)}
    reader = readers.get(path.suffix, pd.read_parquet)
    frame = reader(path)
    missing = [c for c in (id_column, text_column) if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} has no column {missing}; columns are {list(frame.columns)}")

    docs = pd.DataFrame(
        {
            "id": frame[id_column].astype(str).str.strip(),
            "title": frame[title_column].fillna("").astype(str)
            if title_column in frame.columns
            else "",
            "text": frame[text_column].fillna("").astype(str),
        }
    )
    if docs["id"].duplicated().any():
        dupes = docs.loc[docs["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"ids must be unique; repeated: {dupes}")
    if (docs["id"] == "").any() or docs["id"].str.contains(r"\s").any():
        raise ValueError("ids must be non-empty and contain no whitespace")
    return docs


def chunk_text(text: str, chunk_words: int = 300, overlap_words: int = 50) -> list[str]:
    """Overlapping windows of `chunk_words` words; text that fits in one window is kept whole."""
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must be at least 0 and less than chunk_words")
    words = text.split()
    if len(words) <= chunk_words:
        return [text.strip()]
    step = chunk_words - overlap_words
    starts = range(0, len(words) - overlap_words, step)
    return [" ".join(words[i : i + chunk_words]) for i in starts]


def to_chunks(docs: pd.DataFrame, chunk_words: int = 300, overlap_words: int = 50) -> pd.DataFrame:
    """One row per searchable chunk, in the `_id, title, text` layout of corpus.parquet."""
    rows = []
    for doc_id, title, text in zip(docs["id"], docs["title"], docs["text"], strict=True):
        chunks = chunk_text(text, chunk_words, overlap_words)
        for n, chunk in enumerate(chunks, start=1):
            chunk_id = doc_id if len(chunks) == 1 else f"{doc_id}#{n}"
            rows.append({"_id": chunk_id, "title": title, "text": chunk})
    return pd.DataFrame(rows, columns=["_id", "title", "text"])


def build_bm25(corpus: pd.DataFrame, out_dir: Path) -> None:
    """Tokenized exactly as BM25Retriever tokenizes queries (English stopwords removed)."""
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus["text"].tolist(), stopwords="en", show_progress=False))
    bm25_dir = out_dir / "bm25"
    retriever.save(str(bm25_dir))
    (bm25_dir / "doc_ids.txt").write_text("\n".join(corpus["_id"]))


async def embed_corpus(
    texts: list[str], embed_batch: EmbedBatch, batch: int = EMBED_BATCH
) -> np.ndarray:
    """Rows in input order. Blank texts become "[empty document]", which the API accepts."""
    texts = [t.strip() or "[empty document]" for t in texts]
    batches = [texts[i : i + batch] for i in range(0, len(texts), batch)]
    limit = asyncio.Semaphore(EMBED_CONCURRENCY)
    done = 0

    async def one(chunk: list[str]) -> np.ndarray:
        nonlocal done
        async with limit:
            vectors = await embed_batch(chunk)
        done += 1
        print(f"  embedded {done}/{len(batches)} batches", flush=True)
        return vectors

    return np.vstack(await asyncio.gather(*(one(b) for b in batches)))


def openai_embedder(client: AsyncOpenAI) -> EmbedBatch:
    """Embeds with the model the app searches with; waits out rate limits instead of failing."""

    async def embed_batch(texts: list[str]) -> np.ndarray:
        while True:
            try:
                response = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
                return np.array([d.embedding for d in response.data], dtype=np.float32)
            except RateLimitError:
                await asyncio.sleep(RATE_LIMIT_WAIT_S)

    return embed_batch


async def build_indexes(corpus: pd.DataFrame, out_dir: Path, embed_batch: EmbedBatch) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(out_dir / "corpus.parquet", index=False)
    build_bm25(corpus, out_dir)
    embeddings = await embed_corpus(corpus["text"].tolist(), embed_batch)
    np.save(out_dir / "embeddings.npy", embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="the app's data dir, e.g. data")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--chunk-words", type=int, default=300)
    parser.add_argument("--overlap-words", type=int, default=50)
    args = parser.parse_args()

    docs = load_documents(args.input, args.id_column, args.text_column, args.title_column)
    corpus = to_chunks(docs, args.chunk_words, args.overlap_words)
    print(f"{len(docs)} documents -> {len(corpus)} chunks; embedding with {EMBEDDING_MODEL}")

    client = AsyncOpenAI(api_key=get_settings().openai_api_key, max_retries=0)
    asyncio.run(build_indexes(corpus, args.out, openai_embedder(client)))
    print(f"Wrote corpus.parquet, bm25/ and embeddings.npy to {args.out}")


if __name__ == "__main__":
    main()

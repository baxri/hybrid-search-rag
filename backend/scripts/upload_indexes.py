"""One-time: publish the FiQA indexes as a public Hugging Face dataset.

The files are too big for git (embeddings.npy is 338 MB). Log in first with a write token:
    uv run hf auth login
    uv run python -m scripts.upload_indexes --repo <user>/fiqa-hybrid-indexes --src data
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from scripts.download_indexes import INDEX_FILES

DATASET_CARD = """\
---
license: cc-by-sa-4.0
pretty_name: FiQA hybrid search indexes
tags:
- retrieval
- bm25
- embeddings
- finance
---

# FiQA hybrid search indexes

Prebuilt indexes over the [FiQA-2018](https://huggingface.co/datasets/BeIR/fiqa) corpus
(57,638 personal-finance forum posts, via BEIR), used by the
[hybrid-search-rag](https://github.com/baxri/hybrid-search-rag) demo.

| File | Contents |
|---|---|
| `corpus.parquet` | `_id`, `title`, `text` of every post (FiQA corpus, unchanged) |
| `embeddings.npy` | float32 `(57638, 1536)` OpenAI `text-embedding-3-small` vectors, row order = `corpus.parquet` |
| `bm25/` | [bm25s](https://github.com/xhluca/bm25s) index plus `doc_ids.txt` in matching order |

Blank posts (38) were embedded as `"[empty document]"` to keep row order aligned.

FiQA is licensed CC BY-SA 4.0; these derived indexes are shared under the same license.
"""


def upload_indexes(repo: str, src: Path) -> None:
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )
    api.upload_folder(
        folder_path=src,
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=INDEX_FILES,
        commit_message="Upload FiQA hybrid search indexes",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="e.g. someone/fiqa-hybrid-indexes")
    parser.add_argument("--src", type=Path, required=True)
    args = parser.parse_args()
    upload_indexes(args.repo, args.src)


if __name__ == "__main__":
    main()

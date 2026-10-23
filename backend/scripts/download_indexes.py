"""Fetch the prebuilt FiQA indexes from the public Hugging Face dataset (no token needed).

Runs at Docker build time so restarts never re-download:
    python scripts/download_indexes.py --repo <user>/fiqa-hybrid-indexes --dest data
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

INDEX_FILES = ["corpus.parquet", "embeddings.npy", "bm25/*"]


def download_indexes(repo: str, dest: Path) -> None:
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest, allow_patterns=INDEX_FILES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="e.g. someone/fiqa-hybrid-indexes")
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    download_indexes(args.repo, args.dest)


if __name__ == "__main__":
    main()

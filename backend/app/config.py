from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    openai_api_key: str
    cohere_api_key: str
    chat_model: str = "gpt-5.4-mini"
    judge_model: str = "gpt-5.4-mini"  # evals only; same as chat for now (owner decision)

    # Layout: corpus.parquet, embeddings.npy, bm25/
    data_dir: Path = BACKEND_DIR / "data"

    candidate_k: int = 50
    top_k: int = 8
    rrf_k: int = 60
    rerank_model: str = "rerank-v4.0-pro"  # beat -fast by 8 points Hit@8 on the bench

    # The domain: defaults are FiQA's personal-finance forum. For your own documents, see
    # docs/USING_YOUR_DATA.md.
    instructions_file: Path | None = None  # replaces the built-in agent instructions
    search_tool_name: str = "search_finance_posts"
    search_tool_description: str = (
        "Search the FiQA personal-finance forum (BM25 + embeddings, fused and reranked)."
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

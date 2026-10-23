from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    openai_api_key: str
    cohere_api_key: str
    chat_model: str = "gpt-5.4-mini"

    # Layout: corpus.parquet, embeddings.npy, bm25/
    data_dir: Path = BACKEND_DIR / "data"
    hf_repo: str = "baxri/fiqa-hybrid-indexes"

    candidate_k: int = 50
    top_k: int = 8
    rrf_k: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()

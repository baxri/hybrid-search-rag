from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.retrieval.pipeline import HybridPipeline


def create_app(pipeline: HybridPipeline | None = None) -> FastAPI:
    """Build the app. Tests inject a fake pipeline; production loads the real one at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Indexes (~400 MB) load once here, never per request.
        app.state.pipeline = pipeline or HybridPipeline.from_settings(get_settings())
        yield

    app = FastAPI(title="Hybrid Search Chat", lifespan=lifespan)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

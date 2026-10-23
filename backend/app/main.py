from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI


def create_app(pipeline: Any | None = None) -> FastAPI:
    """Build the app. Tests inject a fake pipeline; production builds the real one at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = pipeline
        yield

    app = FastAPI(title="Hybrid Search Chat", lifespan=lifespan)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

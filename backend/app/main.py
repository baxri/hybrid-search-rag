from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.agent import FIQA, Domain, build_agent
from app.config import BACKEND_DIR, get_settings
from app.retrieval.pipeline import HybridPipeline
from app.schemas import ChatRequest
from app.stream import chat_events

FRONTEND_DIR = BACKEND_DIR.parent / "frontend" / "dist"


def create_app(
    pipeline: HybridPipeline | None = None,
    model: Model | str | None = None,
    frontend_dir: Path = FRONTEND_DIR,
    domain: Domain | None = None,
) -> FastAPI:
    """Build the app. Tests inject a fake pipeline and model; production loads real ones at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        chat_model = model
        agent_domain = domain or FIQA
        if pipeline is None or chat_model is None:
            settings = get_settings()
            agent_domain = domain or Domain.from_settings(settings)
            # Pass the key explicitly: settings read backend/.env, which os.environ doesn't see.
            chat_model = chat_model or OpenAIResponsesModel(
                settings.chat_model, provider=OpenAIProvider(api_key=settings.openai_api_key)
            )
        # Indexes (~400 MB) load once here, never per request.
        app.state.pipeline = pipeline or HybridPipeline.from_settings(get_settings())
        app.state.agent = build_agent(chat_model, agent_domain)
        yield

    app = FastAPI(title="Hybrid Search Chat", lifespan=lifespan)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        return StreamingResponse(
            chat_events(body, request.app.state.agent, request.app.state.pipeline),
            media_type="application/x-ndjson",
        )

    if frontend_dir.is_dir():
        _serve_frontend(app, frontend_dir.resolve())

    return app


def _serve_frontend(app: FastAPI, root: Path) -> None:
    """Serve the built React app from the same origin: real files as-is, any other path gets
    index.html so the app's own routing takes over. Registered last, so /api/* wins."""

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend(path: str) -> FileResponse:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404)
        file = (root / path).resolve()
        if file.is_file() and file.is_relative_to(root):
            return FileResponse(file)
        return FileResponse(root / "index.html")


app = create_app()

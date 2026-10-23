from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.agent import build_agent
from app.config import get_settings
from app.retrieval.pipeline import HybridPipeline
from app.schemas import ChatRequest
from app.stream import chat_events


def create_app(pipeline: HybridPipeline | None = None, model: Model | str | None = None) -> FastAPI:
    """Build the app. Tests inject a fake pipeline and model; production loads real ones at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        chat_model = model
        if pipeline is None or chat_model is None:
            settings = get_settings()
            # Pass the key explicitly: settings read backend/.env, which os.environ doesn't see.
            chat_model = chat_model or OpenAIResponsesModel(
                settings.chat_model, provider=OpenAIProvider(api_key=settings.openai_api_key)
            )
        # Indexes (~400 MB) load once here, never per request.
        app.state.pipeline = pipeline or HybridPipeline.from_settings(get_settings())
        app.state.agent = build_agent(chat_model)
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

    return app


app = create_app()

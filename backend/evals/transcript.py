"""Run one question through the exact stream the API serves and keep what the browser would see."""

from pydantic import BaseModel, TypeAdapter
from pydantic_ai import Agent

from app.agent import ChatDeps
from app.retrieval.pipeline import HybridPipeline
from app.schemas import ChatEvent, ChatMessage, ChatRequest, SearchTrace, Source
from app.stream import chat_events

_EVENT = TypeAdapter(ChatEvent)


class Transcript(BaseModel):
    searches: list[SearchTrace] = []
    sources: list[Source] = []  # every post numbered in this turn, as the model saw them
    answer: str = ""
    error: str | None = None


async def collect(
    question: str, agent: Agent[ChatDeps, str], pipeline: HybridPipeline
) -> Transcript:
    request = ChatRequest(messages=[ChatMessage(role="user", content=question)])
    transcript = Transcript()
    finished = False
    async for line in chat_events(request, agent, pipeline):
        event = _EVENT.validate_json(line)
        match event.type:
            case "search_result":
                transcript.searches.append(event.trace)
                transcript.sources.extend(event.sources)
            case "text_delta":
                transcript.answer += event.delta
            case "error":
                transcript.error = event.message
                finished = True
            case "done":
                finished = True
    if not finished:
        transcript.error = "Stream ended without finishing."
    return transcript

"""API and search-trace models shared by the pipeline, agent, stream and frontend."""

from typing import Literal

from pydantic import BaseModel, model_validator

MAX_MESSAGE_CHARS = 4000

StageName = Literal["bm25", "dense", "rrf", "rerank"]


class RankedDoc(BaseModel):
    doc_id: str
    rank: int  # 1-based
    score: float


class Stage(BaseModel):
    name: StageName
    ms: float
    results: list[RankedDoc]


class SearchTrace(BaseModel):
    """What each retrieval stage returned, in order: bm25, dense, rrf, rerank."""

    query: str
    total_ms: float
    stages: list[Stage]


class Source(BaseModel):
    """A post the answer may cite as [n]."""

    n: int
    doc_id: str
    text: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]

    @model_validator(mode="after")
    def _last_message_is_a_real_question(self) -> "ChatRequest":
        if not self.messages:
            raise ValueError("messages must not be empty")
        last = self.messages[-1]
        if last.role != "user":
            raise ValueError("last message must be from the user")
        if not last.content.strip():
            raise ValueError("last message must not be blank")
        if len(last.content) > MAX_MESSAGE_CHARS:
            raise ValueError(f"last message must be at most {MAX_MESSAGE_CHARS} characters")
        return self

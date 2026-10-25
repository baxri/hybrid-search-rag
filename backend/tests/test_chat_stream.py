import json

from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from app.main import create_app
from app.schemas import ChatEvent, SearchResult
from tests.conftest import FakePipeline, scripted_model

ANSWER = "Use a savings account [1]."
QUESTION = {"messages": [{"role": "user", "content": "Where should I keep my emergency fund?"}]}


def post_chat(pipeline: FakePipeline, payload: dict = QUESTION) -> tuple[int, list[dict]]:
    app = create_app(pipeline=pipeline, model=scripted_model(["emergency fund"], ANSWER))
    with TestClient(app) as client:
        response = client.post("/api/chat", json=payload)
    if response.status_code != 200:
        return response.status_code, []
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return 200, [json.loads(line) for line in response.text.splitlines() if line]


def test_stream_event_order():
    _, events = post_chat(FakePipeline())

    types = [e["type"] for e in events]
    assert types[:2] == ["search_started", "search_result"]
    assert set(types[2:-1]) == {"text_delta"}
    assert types[-1] == "done"
    assert "".join(e["delta"] for e in events if e["type"] == "text_delta") == ANSWER
    assert events[0] == {"type": "search_started", "search_id": 1, "query": "emergency fund"}


def test_search_result_carries_sources_and_trace():
    _, events = post_chat(FakePipeline(ranking=["d0", "d3"]))

    result = next(e for e in events if e["type"] == "search_result")
    assert result["search_id"] == 1
    assert [s["name"] for s in result["trace"]["stages"]] == [
        "bm25",
        "dense",
        "rrf",
        "rerank",
    ]
    assert [(s["n"], s["doc_id"]) for s in result["sources"]] == [(1, "d0"), (2, "d3")]
    assert result["sources"][0]["text"].startswith("Keep your emergency fund")


def test_reranker_failure_emits_error_and_closes():
    _, events = post_chat(FakePipeline(error=RuntimeError("cohere is down")))

    assert events[-1] == {
        "type": "error",
        "message": "Something went wrong while answering. Please retry.",
    }
    assert "done" not in [e["type"] for e in events]


def test_empty_message_rejected():
    status, _ = post_chat(FakePipeline(), {"messages": [{"role": "user", "content": "  "}]})

    assert status == 422


def test_last_message_must_be_user():
    payload = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }

    status, _ = post_chat(FakePipeline(), payload)

    assert status == 422


def test_every_line_is_a_typed_chat_event():
    _, events = post_chat(FakePipeline())

    adapter = TypeAdapter(ChatEvent)
    parsed = [adapter.validate_python(e) for e in events]

    assert isinstance(parsed[1], SearchResult)
    assert parsed[1].sources[0].n == 1


def test_first_message_is_passed_to_search_as_the_users_question():
    pipeline = FakePipeline()

    post_chat(pipeline)

    assert pipeline.questions == ["Where should I keep my emergency fund?"]


def test_follow_up_is_not_passed_as_a_question():
    # "What about for kids?" means nothing alone; the agent's standalone rewrite is used instead.
    pipeline = FakePipeline()
    payload = {
        "messages": [
            {"role": "user", "content": "Where should I keep my emergency fund?"},
            {"role": "assistant", "content": "In a savings account [1]."},
            {"role": "user", "content": "What about for kids?"},
        ]
    }

    post_chat(pipeline, payload)

    assert pipeline.questions == [None]

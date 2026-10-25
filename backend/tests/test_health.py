from fastapi.testclient import TestClient

from app.agent import Domain
from app.main import create_app
from tests.conftest import FakePipeline


def test_health_ok():
    with TestClient(create_app(pipeline=FakePipeline(), model="test")) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_startup_builds_real_pipeline_when_none_injected(monkeypatch):
    built = FakePipeline()
    monkeypatch.setattr("app.main.get_settings", lambda: "settings")
    monkeypatch.setattr(
        "app.main.HybridPipeline.from_settings",
        lambda settings: built if settings == "settings" else None,
    )
    domain = Domain(instructions="Answer from the handbook.", tool_name="search_handbook")
    monkeypatch.setattr(
        "app.main.Domain.from_settings", lambda settings: domain if settings == "settings" else None
    )

    app = create_app(model="test")
    with TestClient(app):
        assert app.state.pipeline is built
        assert [t.name for t in app.state.agent.toolsets[0].tools.values()] == ["search_handbook"]

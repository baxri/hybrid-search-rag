from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import FakePipeline


def test_health_ok():
    with TestClient(create_app(pipeline=FakePipeline())) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

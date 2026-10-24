from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import FakePipeline


def built_frontend(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "eval").mkdir()
    (dist / "index.html").write_text("<html>app</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    (dist / "eval" / "results.json").write_text('{"summary": {}}')
    (tmp_path / "secret.txt").write_text("outside dist")
    return dist


def client_for(dist):
    return TestClient(create_app(pipeline=FakePipeline(), model="test", frontend_dir=dist))


def test_spa_fallback(tmp_path):
    with client_for(built_frontend(tmp_path)) as client:
        home = client.get("/")
        any_route = client.get("/some/route")
        health = client.get("/api/health")

    assert home.status_code == 200 and home.text == "<html>app</html>"
    assert any_route.status_code == 200 and any_route.text == "<html>app</html>"
    assert health.json() == {"status": "ok"}


def test_built_files_are_served_as_themselves(tmp_path):
    with client_for(built_frontend(tmp_path)) as client:
        script = client.get("/assets/app.js")
        results = client.get("/eval/results.json")

    assert script.text == "console.log(1)"
    assert results.json() == {"summary": {}}


def test_unknown_api_path_is_a_404_not_the_app(tmp_path):
    with client_for(built_frontend(tmp_path)) as client:
        response = client.get("/api/nope")

    assert response.status_code == 404


def test_cannot_read_files_outside_the_build(tmp_path):
    with client_for(built_frontend(tmp_path)) as client:
        response = client.get("/..%2Fsecret.txt")

    assert "outside dist" not in response.text


def test_no_build_means_api_only(tmp_path):
    with client_for(tmp_path / "missing") as client:
        assert client.get("/").status_code == 404
        assert client.get("/api/health").status_code == 200

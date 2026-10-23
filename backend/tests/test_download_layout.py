from app.config import Settings
from app.retrieval.pipeline import HybridPipeline
from scripts import download_indexes
from tests.fixtures import build_fixture_data


def test_layout_matches_settings(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs) -> str:
        calls.append(kwargs)
        build_fixture_data(kwargs["local_dir"])
        return str(kwargs["local_dir"])

    monkeypatch.setattr(download_indexes, "snapshot_download", fake_snapshot_download)
    dest = tmp_path / "data"

    download_indexes.download_indexes("someone/fiqa-hybrid-indexes", dest)

    assert calls[0]["repo_id"] == "someone/fiqa-hybrid-indexes"
    assert calls[0]["repo_type"] == "dataset"
    settings = Settings(openai_api_key="x", cohere_api_key="x", data_dir=dest)
    pipeline = HybridPipeline.from_settings(settings)
    assert len(pipeline.corpus.ids) == 6

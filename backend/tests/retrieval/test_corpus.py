from app.retrieval.corpus import Corpus
from tests.fixtures import FIXTURE_DOCS, build_fixture_data


def test_corpus_blank_detection(tmp_path):
    corpus = Corpus.load(build_fixture_data(tmp_path))

    assert corpus.is_blank("d4")
    assert not corpus.is_blank("d0")


def test_corpus_text_and_row_order(tmp_path):
    corpus = Corpus.load(build_fixture_data(tmp_path))

    assert corpus.ids == list(FIXTURE_DOCS)
    assert corpus.text("d1") == FIXTURE_DOCS["d1"]

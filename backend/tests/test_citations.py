from app.citations import SourceRegistry, format_for_model, strip_markers
from app.retrieval.corpus import Corpus
from app.schemas import Source
from tests.conftest import canned_trace
from tests.fixtures import FIXTURE_DOCS

CORPUS = Corpus(list(FIXTURE_DOCS), list(FIXTURE_DOCS.values()))


def test_registry_numbers_continue_across_searches():
    registry = SourceRegistry()

    first, _ = registry.add(canned_trace("q1", ["d0", "d1"]), CORPUS)
    second, _ = registry.add(canned_trace("q2", ["d2"]), CORPUS)

    assert [(s.n, s.doc_id) for s in first] == [(1, "d0"), (2, "d1")]
    assert [(s.n, s.doc_id) for s in second] == [(3, "d2")]
    assert first[0].text == FIXTURE_DOCS["d0"]


def test_registry_dedupes_doc_across_searches():
    registry = SourceRegistry()
    registry.add(canned_trace("q1", ["d0", "d1"]), CORPUS)

    this_search, newly_numbered = registry.add(canned_trace("q2", ["d1", "d3"]), CORPUS)

    assert [s.n for s in this_search] == [2, 3]
    assert [(s.n, s.doc_id) for s in newly_numbered] == [(3, "d3")]
    assert len(registry.all) == 3


def test_format_for_model_numbers_and_truncates():
    sources = [Source(n=1, doc_id="a", text="short"), Source(n=2, doc_id="b", text="x" * 2500)]

    text = format_for_model(sources)

    first, second = text.split("\n\n")
    assert first == "[1] short"
    assert second == "[2] " + "x" * 2000


def test_format_for_model_empty():
    assert format_for_model([]) == "No relevant posts found."


def test_strip_markers():
    assert strip_markers("Use CDs [1][2], or HYSA [3, 4].") == "Use CDs, or HYSA."
    assert strip_markers("Keep it liquid [1,2] always.") == "Keep it liquid always."

import paper_research_agent.features.contrast.node as contrast_node
from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import ResearchGap, ResearchState
from paper_research_agent.features.contrast.node import find_gaps
from paper_research_agent.features.contrast.schemas import GapAnalysis


class _FakeStructured:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def invoke(self, messages):
        if self._error is not None:
            raise self._error
        return self._result


class _FakeModel:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def with_structured_output(self, schema):
        return _FakeStructured(result=self._result, error=self._error)


def _patch_llm(monkeypatch, *, result=None, error=None):
    monkeypatch.setattr(
        contrast_node,
        "chat_model_for_tier",
        lambda tier, temperature=0.0: _FakeModel(result=result, error=error),
    )


def _paper(title, url):
    return Paper(title=title, source="arxiv", url=url, abstract="a", year=2024)


def test_discover_mode_when_no_gaps_yet(monkeypatch):
    _patch_llm(
        monkeypatch,
        result=GapAnalysis(
            gaps=[ResearchGap(description="g1", confidence="medium")]
        ),
    )

    state = ResearchState(topic="t", papers=[_paper("P1", "http://1")])
    find_gaps(state)

    assert [(g.description, g.confidence) for g in state.gaps] == [("g1", "medium")]
    assert state.errors == []


def test_update_mode_refines_existing_gaps_with_new_papers(monkeypatch):
    # The new paper strengthens g1: medium -> high.
    _patch_llm(
        monkeypatch,
        result=GapAnalysis(
            gaps=[ResearchGap(description="g1", confidence="high")]
        ),
    )

    state = ResearchState(
        topic="t",
        papers=[_paper("P1", "http://1"), _paper("P2", "http://2")],
        gaps=[ResearchGap(description="g1", confidence="medium")],
        recent_paper_ids=["http://2"],  # P2 is this round's new paper
    )
    find_gaps(state)

    assert [(g.description, g.confidence) for g in state.gaps] == [("g1", "high")]


def test_update_mode_skips_llm_when_no_new_papers(monkeypatch):
    # If the LLM is called this raises; the no-new-papers shortcut must avoid it.
    _patch_llm(monkeypatch, error=AssertionError("LLM should not be called"))

    state = ResearchState(
        topic="t",
        papers=[_paper("P1", "http://1")],
        gaps=[ResearchGap(description="g1", confidence="medium")],
        recent_paper_ids=[],  # nothing new this round
    )
    find_gaps(state)

    # gaps kept unchanged, no error
    assert [(g.description, g.confidence) for g in state.gaps] == [("g1", "medium")]
    assert state.errors == []


def test_contrast_records_error_on_failure(monkeypatch):
    _patch_llm(monkeypatch, error=RuntimeError("boom"))

    state = ResearchState(topic="t", papers=[_paper("P1", "http://1")])
    find_gaps(state)

    assert state.gaps == []  # discover failed -> no gaps
    assert len(state.errors) == 1
    assert state.errors[0].startswith("contrast failed:")


def test_no_papers_returns_unchanged(monkeypatch):
    _patch_llm(monkeypatch, error=AssertionError("LLM should not be called"))

    state = ResearchState(topic="t")  # no papers
    find_gaps(state)

    assert state.gaps == []
    assert state.errors == []

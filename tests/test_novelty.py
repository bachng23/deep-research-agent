import paper_research_agent.features.novelty.node as novelty_node
from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import ResearchGap, ResearchState
from paper_research_agent.features.novelty import score_novelty
from paper_research_agent.features.novelty.schemas import NoveltyAssessment


class _FakeStructured:
    """Stands in for model.with_structured_output(...): .invoke returns
    a fixed result, or raises if one was provided."""

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
    """Swap the node's chat_model_for_tier so no real LLM is called."""
    monkeypatch.setattr(
        novelty_node,
        "chat_model_for_tier",
        lambda tier, temperature=0.0: _FakeModel(result=result, error=error),
    )


def _state_with_papers(**kwargs) -> ResearchState:
    return ResearchState(
        topic="retrieval augmented generation",
        user_idea="hierarchical chunking for better recall",
        papers=[
            Paper(title="RAG basics", source="arxiv", abstract="a", year=2023),
        ],
        gaps=[ResearchGap(description="no long-doc benchmark", confidence="high")],
        **kwargs,
    )


def test_score_novelty_fills_state(monkeypatch):
    assessment = NoveltyAssessment(
        score=72,
        reasoning="Partially explored; adds a new angle.",
        overlapping_papers=["RAG basics"],
    )
    _patch_llm(monkeypatch, result=assessment)

    out = score_novelty(_state_with_papers())

    assert out.novelty_score == 72
    assert out.novelty_reasoning == "Partially explored; adds a new angle."
    assert out.overlapping_papers == ["RAG basics"]
    assert out.errors == []


def test_score_novelty_skips_without_idea(monkeypatch):
    # If the LLM were called this would blow up; it must not be reached.
    _patch_llm(monkeypatch, error=AssertionError("LLM should not be called"))

    state = _state_with_papers()
    state.user_idea = None

    out = score_novelty(state)

    assert out.novelty_score is None
    assert out.errors == []


def test_score_novelty_skips_without_papers(monkeypatch):
    _patch_llm(monkeypatch, error=AssertionError("LLM should not be called"))

    state = _state_with_papers()
    state.papers = []

    out = score_novelty(state)

    assert out.novelty_score is None
    assert out.errors == []


def test_score_novelty_records_error_on_failure(monkeypatch):
    _patch_llm(monkeypatch, error=RuntimeError("boom"))

    out = score_novelty(_state_with_papers())

    assert out.novelty_score is None
    assert len(out.errors) == 1
    assert out.errors[0].startswith("novelty failed:")

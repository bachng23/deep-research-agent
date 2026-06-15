import paper_research_agent.features.writing.node as writing_node
from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import ResearchGap, ResearchState
from paper_research_agent.features.writing import write_report
from paper_research_agent.features.writing.node import _format_references


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeModel:
    """Stands in for chat_model_for_tier(...): .invoke returns a message
    with fixed content, or raises if an error was provided."""

    def __init__(self, content=None, error=None):
        self._content = content
        self._error = error

    def invoke(self, messages):
        if self._error is not None:
            raise self._error
        return _FakeMessage(self._content)


def _patch_llm(monkeypatch, *, content=None, error=None):
    monkeypatch.setattr(
        writing_node,
        "chat_model_for_tier",
        lambda tier, temperature=0.0: _FakeModel(content=content, error=error),
    )


def _state_with_papers(**kwargs) -> ResearchState:
    return ResearchState(
        topic="retrieval augmented generation",
        user_idea="hierarchical chunking",
        papers=[
            Paper(title="RAG basics", source="arxiv", abstract="a", year=2023),
            Paper(title="Long-doc QA", source="openalex", abstract="b", year=2024),
        ],
        gaps=[ResearchGap(description="no long-doc benchmark", confidence="high")],
        **kwargs,
    )


def test_write_report_assembles_body_and_references(monkeypatch):
    _patch_llm(monkeypatch, content="## Overview\nGrounded claim [1].")

    out = write_report(_state_with_papers())

    assert out.report_markdown is not None
    # LLM-written body is preserved
    assert "## Overview" in out.report_markdown
    assert "Grounded claim [1]." in out.report_markdown
    # References section is appended deterministically, numbered to match
    assert "## References" in out.report_markdown
    assert "[1] RAG basics" in out.report_markdown
    assert "[2] Long-doc QA" in out.report_markdown
    assert out.errors == []


def test_write_report_skips_without_papers(monkeypatch):
    _patch_llm(monkeypatch, error=AssertionError("LLM should not be called"))

    state = _state_with_papers()
    state.papers = []

    out = write_report(state)

    assert out.report_markdown is None
    assert out.errors == []


def test_write_report_records_error_on_failure(monkeypatch):
    _patch_llm(monkeypatch, error=RuntimeError("boom"))

    out = write_report(_state_with_papers())

    assert out.report_markdown is None
    assert len(out.errors) == 1
    assert out.errors[0].startswith("writing failed:")


def test_format_references_truncates_authors():
    papers = [
        Paper(
            title="Many authors",
            source="arxiv",
            year=2022,
            authors=["A", "B", "C", "D"],
            url="http://x",
        ),
    ]

    refs = _format_references(papers)

    assert refs.startswith("## References")
    assert "[1] Many authors" in refs
    assert "A, B, C et al." in refs
    assert "http://x" in refs

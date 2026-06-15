import paper_research_agent.features.planning.node as planning_node
from paper_research_agent.core.state import ResearchState
from paper_research_agent.features.planning.node import _clean_queries, plan_queries
from paper_research_agent.features.planning.schemas import QueryPlan


class _FakeStructured:
    def __init__(self, queries=None, error=None):
        self._queries = queries
        self._error = error

    def invoke(self, messages):
        if self._error is not None:
            raise self._error
        return QueryPlan(queries=self._queries)


class _FakeModel:
    def __init__(self, queries=None, error=None):
        self._queries = queries
        self._error = error

    def with_structured_output(self, schema):
        return _FakeStructured(queries=self._queries, error=self._error)


def _patch_llm(monkeypatch, *, queries=None, error=None):
    monkeypatch.setattr(
        planning_node,
        "chat_model_for_tier",
        lambda tier, temperature=0.0: _FakeModel(queries=queries, error=error),
    )


def test_plan_initial_from_topic(monkeypatch):
    _patch_llm(monkeypatch, queries=["q1", "q2"])

    state = ResearchState(topic="rag long documents")
    plan_queries(state)

    assert state.search_queries == ["q1", "q2"]
    assert state.errors == []


def test_plan_followup_replans_on_open_gaps(monkeypatch):
    _patch_llm(monkeypatch, queries=["gap query 1", "gap query 2"])

    # round 2: queries already exist, but open gaps force a re-plan
    state = ResearchState(
        topic="rag long documents",
        search_queries=["old query"],
        open_gaps=["hierarchical chunking before retrieval"],
    )
    plan_queries(state)

    assert state.search_queries == ["gap query 1", "gap query 2"]


def test_plan_keeps_pre_supplied_queries(monkeypatch):
    # If the LLM were called this would raise; the guard must prevent it.
    _patch_llm(monkeypatch, error=AssertionError("LLM should not be called"))

    state = ResearchState(topic="t", search_queries=["user query"])
    plan_queries(state)

    assert state.search_queries == ["user query"]
    assert state.errors == []


def test_plan_falls_back_to_topic_on_error(monkeypatch):
    _patch_llm(monkeypatch, error=RuntimeError("boom"))

    state = ResearchState(topic="my topic")
    plan_queries(state)

    assert state.search_queries == ["my topic"]
    assert len(state.errors) == 1
    assert state.errors[0].startswith("planning failed:")


def test_clean_queries_dedups_and_normalizes():
    cleaned = _clean_queries(
        ["  RAG  long  docs ", "rag long docs", "", "another query"],
        fallback="topic",
    )

    # whitespace-normalized, case-insensitive dedup, blanks dropped
    assert cleaned == ["RAG long docs", "another query"]


def test_clean_queries_falls_back_when_empty():
    assert _clean_queries(["", "   "], fallback="topic") == ["topic"]

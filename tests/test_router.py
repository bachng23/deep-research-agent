import paper_research_agent.agent.router as router_mod
from paper_research_agent.agent.router import Intent, route


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
        return _FakeStructured(self._result, self._error)


def _patch(monkeypatch, *, result=None, error=None):
    monkeypatch.setattr(
        router_mod,
        "chat_model_for_tier",
        lambda tier, **kwargs: _FakeModel(result=result, error=error),
    )


def test_route_research(monkeypatch):
    _patch(monkeypatch, result=Intent(action="research"))
    assert route("graph neural networks for drug discovery") == "research"


def test_route_qa(monkeypatch):
    _patch(monkeypatch, result=Intent(action="qa"))
    assert route("what causes hallucination in RAG?") == "qa"


def test_route_falls_back_to_qa_on_error(monkeypatch):
    _patch(monkeypatch, error=RuntimeError("boom"))
    assert route("anything") == "qa"  # safe (cheap) default when the LLM fails

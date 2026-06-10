from paper_research_agent.agent.graph import run_research
from paper_research_agent.core.models import Paper
from paper_research_agent.features.fetching import node as fetching_node


class StubProvider:
    name = "stub"

    def search(self, query: str, max_results: int | None = None) -> list[Paper]:
        return [
            Paper(
                title=f"Paper about {query}",
                source="arxiv",
                abstract="An abstract.",
                year=2024,
            )
        ]


def test_run_research_end_to_end_with_stub_provider(monkeypatch):
    monkeypatch.setattr(
        fetching_node, "default_providers", lambda: [StubProvider()]
    )

    state = run_research(topic="retrieval augmented generation")

    assert state.search_queries == ["retrieval augmented generation"]
    assert len(state.papers) == 1
    assert state.papers[0].title == "Paper about retrieval augmented generation"
    assert state.tool_call_count == 1

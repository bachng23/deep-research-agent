import pytest

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


@pytest.mark.integration
def test_run_research_end_to_end_with_stub_provider(monkeypatch):
    monkeypatch.setattr(
        fetching_node, "default_providers", lambda: [StubProvider()]
    )

    state = run_research(topic="retrieval augmented generation")

    # Planner produced at least one query (real LLM output varies)
    assert state.search_queries
    # Fetch ran through the stub provider
    assert state.papers
    assert state.tool_call_count >= 1
    # Writer closed the loop
    assert state.report_markdown is not None

"""End-to-end test of the Phase 2 research loop with every LLM mocked.

Forces exactly two rounds: round 1 yields a medium-confidence gap (stays open ->
loop continues), round 2 yields a high-confidence gap (closed -> loop stops and
the reporting tail runs).
"""

import paper_research_agent.features.contrast.node as contrast_node
import paper_research_agent.features.coverage.node as coverage_node
import paper_research_agent.features.novelty.node as novelty_node
import paper_research_agent.features.planning.node as planning_node
import paper_research_agent.features.writing.node as writing_node
from paper_research_agent.agent.graph import run_research
from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import ResearchGap
from paper_research_agent.features.contrast.schemas import GapAnalysis
from paper_research_agent.features.coverage.schemas import CoverageJudgment
from paper_research_agent.features.fetching import node as fetching_node
from paper_research_agent.features.novelty.schemas import NoveltyAssessment
from paper_research_agent.features.planning.schemas import QueryPlan


class _Message:
    def __init__(self, content):
        self.content = content


def _sequencing_factory(results):
    """Returns a chat_model_for_tier replacement whose successive .invoke calls
    (across rounds) return results[0], results[1], ... then repeat the last."""
    counter = {"i": 0}

    def next_result():
        idx = min(counter["i"], len(results) - 1)
        counter["i"] += 1
        return results[idx]

    class _Structured:
        def invoke(self, messages):
            return next_result()

    class _Model:
        def with_structured_output(self, schema):
            return _Structured()

        def invoke(self, messages):  # writing node path (no structured output)
            return next_result()

    def factory(tier, temperature=0.0):
        return _Model()

    return factory


class _StubProvider:
    name = "stub"

    def search(self, query: str, max_results=None) -> list[Paper]:
        return [
            Paper(
                title=f"Paper for {query}",
                source="arxiv",
                url=f"http://example/{query}",
                abstract="Abstract sentence one. Abstract sentence two.",
                year=2024,
            )
        ]


def test_loop_runs_two_rounds_then_reports(monkeypatch):
    monkeypatch.setattr(
        fetching_node, "default_providers", lambda: [_StubProvider()]
    )

    # Planner: distinct queries per round.
    monkeypatch.setattr(
        planning_node,
        "chat_model_for_tier",
        _sequencing_factory(
            [
                QueryPlan(queries=["round one query"]),
                QueryPlan(queries=["round two gap query"]),
            ]
        ),
    )

    # Contrast: round 1 medium (open) -> loop; round 2 high (closed) -> stop.
    monkeypatch.setattr(
        contrast_node,
        "chat_model_for_tier",
        _sequencing_factory(
            [
                GapAnalysis(
                    gaps=[ResearchGap(description="open gap", confidence="medium")]
                ),
                GapAnalysis(
                    gaps=[ResearchGap(description="closed gap", confidence="high")]
                ),
            ]
        ),
    )

    monkeypatch.setattr(
        novelty_node,
        "chat_model_for_tier",
        _sequencing_factory(
            [NoveltyAssessment(score=80, reasoning="novel enough")]
        ),
    )

    monkeypatch.setattr(
        writing_node,
        "chat_model_for_tier",
        _sequencing_factory([_Message("## Overview\nReport body [1].")]),
    )

    # Coverage judge always says "not sufficient" so the deterministic open-gap
    # signal drives the loop (keeps this test focused on loop mechanics).
    monkeypatch.setattr(
        coverage_node,
        "chat_model_for_tier",
        _sequencing_factory(
            [CoverageJudgment(sufficient=False, reasoning="keep going")]
        ),
    )

    state = run_research(topic="long-document RAG", user_idea="hierarchical chunking")

    # Ran exactly two rounds.
    assert state.iteration == 2
    assert len(state.round_logs) == 2

    # Round 2 queries differ from round 1 (follow-up targeted the open gap).
    assert state.round_logs[0].queries != state.round_logs[1].queries
    assert state.round_logs[0].queries == ["round one query"]
    assert state.round_logs[1].queries == ["round two gap query"]

    # Loop stopped because gaps closed, not because of an error.
    assert state.open_gaps == []
    assert state.errors == []

    # Reporting tail ran once after the loop.
    assert state.novelty_score == 80
    assert state.report_markdown is not None
    assert "## References" in state.report_markdown

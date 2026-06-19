from __future__ import annotations

from collections.abc import Iterator

from langgraph.graph import END, StateGraph

from paper_research_agent.config import get_settings
from paper_research_agent.core.state import ResearchState
from paper_research_agent.features.conflicts import find_conflicts
from paper_research_agent.features.contrast import find_gaps, rank_gaps, refine_gaps
from paper_research_agent.features.coverage import (
    assess_coverage,
    judge_coverage,
    should_continue,
)
from paper_research_agent.features.fetching import fetch_papers
from paper_research_agent.features.novelty import score_novelty
from paper_research_agent.features.planning import plan_queries
from paper_research_agent.features.reading import read_papers
from paper_research_agent.features.writing import write_report
from paper_research_agent.memory import recall_prior, remember_result


def _route_after_assessment(state: ResearchState) -> str:
    "Conditional edege: loop back for another round, or move on to reporting."
    return "continue" if should_continue(state) else "finish"


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("plan_queries", plan_queries)
    graph.add_node("fetch_papers", fetch_papers)
    graph.add_node("find_gaps", find_gaps)
    graph.add_node("rank_gaps", rank_gaps)
    graph.add_node("find_conflicts", find_conflicts)
    graph.add_node("assess_coverage", assess_coverage)
    graph.add_node("judge_coverage", judge_coverage)
    graph.add_node("score_novelty", score_novelty)
    graph.add_node("write_report", write_report)
    graph.add_node("read_papers", read_papers)
    graph.add_node("refine_gaps", refine_gaps)
    graph.add_node("recall_prior", recall_prior)
    graph.add_node("remember_result", remember_result)

    graph.set_entry_point("recall_prior")

    # The research loop
    graph.add_edge("recall_prior", "plan_queries")
    graph.add_edge("plan_queries", "fetch_papers")
    graph.add_edge("fetch_papers", "find_gaps")
    graph.add_edge("find_gaps", "assess_coverage")
    graph.add_edge("assess_coverage", "judge_coverage")

    # Decision point: keep searching or finish
    graph.add_conditional_edges(
        "judge_coverage",
        _route_after_assessment,
        {
            "continue": "plan_queries",  # loop back to plan_queries
            "finish": "read_papers",
        },
    )

    graph.add_edge("read_papers", "refine_gaps")
    graph.add_edge("refine_gaps", "rank_gaps")
    graph.add_edge("rank_gaps", "find_conflicts")
    graph.add_edge("find_conflicts", "score_novelty")
    graph.add_edge("score_novelty", "write_report")
    graph.add_edge("write_report", "remember_result")
    graph.add_edge("remember_result", END)

    return graph.compile()


def run_research(
    topic: str,
    user_idea: str | None = None,
    *,
    read_full_text: bool = False,
    max_iterations: int = 3,
    use_memory=None,
) -> ResearchState:
    if use_memory is None:
        use_memory = get_settings().use_memory

    app = build_graph()

    initial_state = ResearchState(
        topic=topic,
        user_idea=user_idea,
        read_full_text=read_full_text,
        max_iterations=max_iterations,
        use_memory=use_memory,
    )

    result = app.invoke(initial_state)
    return ResearchState.model_validate(result)


def stream_research(
    topic: str,
    user_idea: str | None = None,
    *,
    read_full_text: bool = False,
    max_iterations: int = 3,
    use_memory=None,
) -> Iterator[tuple[str, object]]:
    """Stream a research run for a live UI.

    Yields ("node", <node_name>, <ResearchState after that node>) per step,
    then ("done", <final ResearchState>). Nodes return the whole state, so the
    "updates" delta is the full state -- handy for showing live counts."""
    app = build_graph()

    if use_memory is None:
        use_memory = get_settings().use_memory

    initial_state = ResearchState(
        topic=topic,
        user_idea=user_idea,
        read_full_text=read_full_text,
        max_iterations=max_iterations,
        use_memory=use_memory,
    )

    final: ResearchState | None = None
    for chunk in app.stream(initial_state, stream_mode="updates"):
        for node_name, node_state in chunk.items():
            state = (
                node_state
                if isinstance(node_state, ResearchState)
                else ResearchState.model_validate(node_state)
            )
            final = state
            yield ("node", node_name, state)

    if final is not None:
        yield ("done", final)

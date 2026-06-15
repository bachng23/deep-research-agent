from __future__ import annotations

from paper_research_agent.core.state import ResearchState, RoundLog


def assess_coverage(state: ResearchState) -> ResearchState:
    state.iteration += 1

    open_gaps = [g.description for g in state.gaps if g.confidence != "high"]
    closed_gaps = len(state.gaps) - len(open_gaps)
    state.open_gaps = open_gaps

    prev_total = state.round_logs[-1].total_papers if state.round_logs else 0
    state.round_logs.append(
        RoundLog(
            iteration=state.iteration,
            queries=list(state.search_queries),
            new_papers=len(state.papers) - prev_total,
            total_papers=len(state.papers),
            open_gaps=len(open_gaps),
            closed_gaps=closed_gaps,
        )
    )

    return state


def should_continue(state: ResearchState) -> bool:
    "Stop criteria: loop only while gaps remain open AND under the ceiling."
    return bool(state.open_gaps) and state.iteration < state.max_iterations

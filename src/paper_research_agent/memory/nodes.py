from __future__ import annotations

from paper_research_agent.config import get_settings
from paper_research_agent.core.state import ResearchState
from paper_research_agent.memory.results import ResultMemory


def _result_path(state: ResearchState) -> str | None:
    s = get_settings()
    return f"{s.memory_dir}/results.db" if state.use_memory else None


def recall_prior(state: ResearchState) -> ResearchState:
    "Start of run: pull gap descriptions from earlier related runs."
    state.recalled_gaps = ResultMemory(_result_path(state)).recall(state.topic)
    return state


def remember_result(state: ResearchState) -> ResearchState:
    "End of run: persist this run's topic -> gaps for future recall."
    ResultMemory(_result_path(state)).remember(state)
    return state

from __future__ import annotations

from paper_research_agent.config import get_settings
from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import ResearchState
from paper_research_agent.features.fetching.dedup import deduplicate_papers, paper_id
from paper_research_agent.features.fetching.mcp_search import search_via_tool_agent
from paper_research_agent.features.fetching.ranking import rank_papers
from paper_research_agent.features.fetching.service import (
    default_providers,
    fetch_papers_for_queries,
)


def fetch_papers(state: ResearchState) -> ResearchState:
    queries = state.search_queries or [state.topic]

    if state.use_tool_agent:
        try:
            new_papers, n_calls = search_via_tool_agent(queries)
        except Exception as e:
            state.errors.append(f"tool-agent fetch failed: {e}")
            new_papers, n_calls = [], 0
        if not new_papers:
            new_papers, n_calls = _fetch_deterministic(state, queries)
    else:
        new_papers, n_calls = _fetch_deterministic(state, queries)

    new_papers = deduplicate_papers(new_papers)
    unseen = _select_unseen(new_papers, state.seen_paper_ids)
    unseen = rank_papers(unseen)[: get_settings().max_new_papers_per_round]

    state.recent_paper_ids = [paper_id(p) for p in unseen]
    state.papers = rank_papers(state.papers + unseen)
    state.seen_paper_ids.extend(paper_id(p) for p in unseen)
    state.tool_call_count += n_calls

    return state


def _fetch_deterministic(
    state: ResearchState, queries: list[str]
) -> tuple[list[Paper], int]:
    result = fetch_papers_for_queries(queries, providers=default_providers())
    state.errors.extend(state.errors)

    return result.papers, result.successful_calls


def _select_unseen(papers: list[Paper], seen_ids: list[str]) -> list[Paper]:
    seen = set(seen_ids)
    unseen: list[Paper] = []

    for paper in papers:
        pid = paper_id(paper)
        if pid in seen:
            continue
        seen.add(pid)  # also guards against dups within this same batch
        unseen.append(paper)

    return unseen

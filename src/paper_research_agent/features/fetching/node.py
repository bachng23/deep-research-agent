from __future__ import annotations

from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import ResearchState
from paper_research_agent.features.fetching.dedup import paper_id
from paper_research_agent.features.fetching.ranking import rank_papers
from paper_research_agent.features.fetching.service import (
    default_providers,
    fetch_papers_for_queries,
)


def fetch_papers(state: ResearchState) -> ResearchState:
    queries = state.search_queries or [state.topic]
    providers = default_providers()

    result = fetch_papers_for_queries(queries, providers=providers)

    state.errors.extend(result.errors)
    state.tool_call_count += result.successful_calls

    new_papers = _select_unseen(result.papers, state.seen_paper_ids)

    # Accumulate across rounds, then re-rank the combined set.
    state.papers = rank_papers(state.papers + new_papers)
    state.seen_paper_ids.extend(paper_id(p) for p in new_papers)

    return state


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

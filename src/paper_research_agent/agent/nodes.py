from __future__ import annotations

from difflib import SequenceMatcher

from paper_research_agent.state import Paper, ResearchState
from paper_research_agent.tools import search_arxiv, search_openalex


def fetch_papers(state: ResearchState) -> ResearchState:
    queries = state.search_queries or [state.topic]

    papers: list[Paper] = []

    for query in queries:
        papers.extend(search_arxiv(query=query))
        papers.extend(search_openalex(query=query))

    ranked_papers = rank_papers(deduplicate_papers(papers))

    state.papers = ranked_papers[:20]
    state.tool_call_count += len(queries) * 2

    return state


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    unique: list[Paper] = []

    for paper in papers:
        if not _is_duplicate(paper, unique):
            unique.append(paper)

    return unique


def rank_papers(papers: list[Paper]) -> list[Paper]:
    return sorted(
        papers,
        key=lambda paper: (
            paper.abstract is not None,
            paper.citation_count or 0,
            paper.year or 0,
        ),
        reverse=True,
    )


def _is_duplicate(paper: Paper, existing_papers: list[Paper]) -> bool:
    normalized_title = _normalize_title(paper.title)

    for existing in existing_papers:
        existing_title = _normalize_title(existing.title)

        if normalized_title == existing_title:
            return True

        similarity = SequenceMatcher(None, normalized_title, existing_title).ratio()
        if similarity >= 0.90:
            return True

    return False


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().strip().split())

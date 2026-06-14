from paper_research_agent.features.fetching.dedup import deduplicate_papers
from paper_research_agent.features.fetching.node import fetch_papers
from paper_research_agent.features.fetching.ranking import rank_papers
from paper_research_agent.features.fetching.service import (
    FetchResult,
    fetch_papers_for_queries,
)

__all__ = [
    "FetchResult",
    "deduplicate_papers",
    "fetch_papers",
    "fetch_papers_for_queries",
    "rank_papers",
]

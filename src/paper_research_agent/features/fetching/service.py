from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import dataclasses

from paper_research_agent.core.models import Paper
from paper_research_agent.features.fetching.dedup import deduplicate_papers
from paper_research_agent.features.fetching.ranking import rank_papers
from paper_research_agent.providers import (
    ArxivProvider,
    OpenAlexProvider,
    PaperProvider,
)


@dataclass(frozen=True)
class FetchResult:
    papers: list[Paper] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    successful_calls: int = 0


def default_providers() -> list[PaperProvider]:
    return [ArxivProvider(), OpenAlexProvider()]


def fetch_papers_for_queries(
    queries: Iterable[str],
    *,
    providers: list[PaperProvider] | None = None,
    max_results_per_provider: int | None = None,
    limit: int = 20,
) -> FetchResult:
    active_providers = providers or default_providers()
    papers: list[Paper] = []
    errors: list[str] = []
    successful_calls = 0

    for query in queries:
        for provider in active_providers:
            try:
                results = provider.search(query, max_results=max_results_per_provider)
            except Exception as e:
                errors.append(f"{provider.name} failed for query {query!r}: {e}")
                continue
            papers.extend(results)
            successful_calls += 1

    papers = deduplicate_papers(papers)
    papers = rank_papers(papers)

    return FetchResult(
        papers=papers[:limit], errors=errors, successful_calls=successful_calls
    )

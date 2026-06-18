from __future__ import annotations

import arxiv

from paper_research_agent.config import get_settings
from paper_research_agent.core.models import Paper


class ArxivProvider:
    name = "arxiv"

    def search(self, query: str, max_results: int | None = None) -> list[Paper]:
        settings = get_settings()
        limit = max_results or settings.arxiv_max_results

        client = arxiv.Client()

        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: list[Paper] = []

        for result in client.results(search=search):
            year = result.published.year if result.published else None

            papers.append(
                Paper(
                    title=result.title.strip(),
                    authors=[author.name for author in result.authors],
                    year=year,
                    abstract=(result.summary or "").strip(),
                    url=result.entry_id,
                    source="arxiv",
                    citation_count=None,
                    pdf_url=result.pdf_url,
                )
            )

        return papers

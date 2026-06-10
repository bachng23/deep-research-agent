"""Hits the real ArXiv and OpenAlex APIs. Run with: pytest -m integration"""

import pytest

from paper_research_agent.providers import ArxivProvider, OpenAlexProvider

pytestmark = pytest.mark.integration


def test_arxiv_provider_returns_papers():
    papers = ArxivProvider().search("retrieval augmented generation", max_results=2)

    assert papers
    assert all(paper.source == "arxiv" for paper in papers)
    assert all(paper.title for paper in papers)


def test_openalex_provider_returns_papers():
    papers = OpenAlexProvider().search("retrieval augmented generation", max_results=2)

    assert papers
    assert all(paper.source == "openalex" for paper in papers)
    assert all(paper.title for paper in papers)

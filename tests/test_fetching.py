from paper_research_agent.core.models import Paper
from paper_research_agent.features.fetching import deduplicate_papers, rank_papers


def test_duplicate_papers_by_similar_title():
    papers = [
        Paper(
            title="Retrieval Augmented Generation for Long Documents",
            source="arxiv",
        ),
        Paper(
            title="retrieval augmented generation for long documents",
            source="openalex",
        ),
    ]

    result = deduplicate_papers(papers)

    assert len(result) == 1


def test_rank_papers_prefers_abstract_citations_and_year():
    papers = [
        Paper(
            title="Old Paper",
            source="openalex",
            abstract=None,
            citation_count=100,
            year=2020,
        ),
        Paper(
            title="New Cited Paper",
            source="openalex",
            abstract="Useful abstract",
            citation_count=50,
            year=2024,
        ),
    ]

    result = rank_papers(papers)

    assert result[0].title == "New Cited Paper"

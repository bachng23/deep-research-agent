from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import ResearchGap, ResearchState
from paper_research_agent.eval.golden import GoldenCase
from paper_research_agent.eval.metrics import (
    citation_coverage,
    gap_keyword_recall,
    grounded_in_fulltext,
    paper_recall,
    score_case,
)


def _paper(title: str, *, excerpt: str | None = None) -> Paper:
    return Paper(
        title=title,
        authors=["A"],
        abstract="abs",
        source="arxiv",
        full_text_excerpt=excerpt,
    )


def _state(**kw) -> ResearchState:
    return ResearchState(topic="t", **kw)


# --- grounded_in_fulltext ------------------------------------------------------


def test_grounded_matches_quote_despite_line_wrapping():
    "PDF/markdown wrap sentences; the model quotes them with single spaces."
    excerpt = "[Results] Our method improves recall\nby 12 points on the long-document\nsplit."
    state = _state(
        papers=[_paper("P", excerpt=excerpt)],
        gaps=[
            ResearchGap(
                description="g",
                evidence_quotes=["Our method improves recall by 12 points"],
            )
        ],
    )
    assert grounded_in_fulltext(state) == 1.0


def test_grounded_does_not_forgive_paraphrase():
    state = _state(
        papers=[_paper("P", excerpt="[Results] Our method improves recall by 12 points.")],
        gaps=[ResearchGap(description="g", evidence_quotes=["recall went up a lot"])],
    )
    assert grounded_in_fulltext(state) == 0.0


def test_grounded_is_none_when_nothing_was_read():
    "No excerpt -> unmeasurable. Must not read as 0% grounded."
    state = _state(
        papers=[_paper("P")],
        gaps=[ResearchGap(description="g", evidence_quotes=["q"])],
    )
    assert grounded_in_fulltext(state) is None


def test_grounded_is_none_when_no_gap_carries_a_quote():
    state = _state(
        papers=[_paper("P", excerpt="[Results] text")],
        gaps=[ResearchGap(description="g")],
    )
    assert grounded_in_fulltext(state) is None


def test_grounded_counts_gaps_not_quotes():
    "One grounded quote is enough for its gap; the denominator is quoted gaps."
    state = _state(
        papers=[_paper("P", excerpt="[Results] alpha beta gamma")],
        gaps=[
            ResearchGap(description="g1", evidence_quotes=["alpha beta", "not present"]),
            ResearchGap(description="g2", evidence_quotes=["also missing"]),
        ],
    )
    assert grounded_in_fulltext(state) == 0.5


# --- citation_coverage ---------------------------------------------------------


def test_citation_coverage_reads_inline_citations_only():
    "Reference-list numbering must not be mistaken for inline citations."
    md = "## Landscape\nClaim [1] and claim [2].\n\n## References\n[1] A  \n[2] B"
    state = _state(papers=[_paper("A"), _paper("B")], report_markdown=md)
    assert citation_coverage(state) == 1.0


def test_citation_coverage_flags_unresolvable_citation():
    md = "## Landscape\nClaim [1] and claim [9].\n\n## References\n[1] A  \n[2] B"
    state = _state(papers=[_paper("A"), _paper("B")], report_markdown=md)
    assert citation_coverage(state) == 0.5


def test_citation_coverage_ignores_the_references_section():
    "A report citing nothing inline scores None, even though references exist."
    md = "## Overview\nNo citations here.\n\n## References\n[1] A  \n[2] B"
    state = _state(papers=[_paper("A"), _paper("B")], report_markdown=md)
    assert citation_coverage(state) is None


def test_citation_coverage_is_none_without_a_report():
    assert citation_coverage(_state()) is None


# --- recall --------------------------------------------------------------------


def test_paper_recall_is_case_insensitive_substring():
    state = _state(papers=[_paper("LoRA: Low-Rank Adaptation"), _paper("Other")])
    case = GoldenCase(topic="t", expected_papers=["lora", "QLoRA"])
    assert paper_recall(state, case) == 0.5


def test_recall_is_none_without_labels():
    state = _state(papers=[_paper("X")], gaps=[ResearchGap(description="d")])
    case = GoldenCase(topic="t")
    assert paper_recall(state, case) is None
    assert gap_keyword_recall(state, case) is None


def test_gap_keyword_recall_over_descriptions():
    state = _state(
        gaps=[
            ResearchGap(description="No evaluation on tables"),
            ResearchGap(description="Long document handling is untested"),
        ]
    )
    case = GoldenCase(topic="t", expected_gap_keywords=["table", "long document", "cost"])
    assert round(gap_keyword_recall(state, case), 2) == 0.67


# --- score_case ----------------------------------------------------------------


def test_score_case_propagates_none_rather_than_zero_or_one():
    state = _state(papers=[_paper("P")], gaps=[ResearchGap(description="g")])
    row = score_case(state, GoldenCase(topic="t"))
    assert row["grounded_in_fulltext"] is None
    assert row["citation_coverage"] is None
    assert row["paper_recall"] is None
    assert row["gap_keyword_recall"] is None
    assert row["papers"] == 1


def test_golden_case_min_gaps_defaults_to_one():
    "Regression: the field was declared with a typo'd kwarg and became required."
    assert GoldenCase(topic="t").min_gaps == 1

from __future__ import annotations

import re

from paper_research_agent.core.state import ResearchState
from paper_research_agent.eval.golden import GoldenCase


def operational(state: ResearchState) -> dict:
    return {
        "papers": len(state.papers),
        "gaps": len(state.gaps),
        "conflicts": len(state.conflicts),
        "rounds": state.iteration,
        "tool_calls": state.tool_call_count,
        "provider_errors": len(state.errors),
    }


def _normalize(text: str) -> str:
    """Collapse whitespace for quote matching.

    Excerpts keep the raw line breaks of the source (PDF extraction and arXiv
    markdown both wrap mid-sentence); a model quoting that sentence emits it
    with single spaces. Comparing the two verbatim fails on formatting alone,
    so both sides are collapsed before matching. Wording is still compared
    exactly -- this forgives layout, not paraphrase.
    """
    return " ".join(text.split()).lower()


def grounded_in_fulltext(state: ResearchState) -> float | None:
    """Fraction of quote-bearing gaps whose evidence quote appears in a full-text
    excerpt. None when nothing was measurable (no excerpts read, or no gap
    carried a quote) -- that is missing data, not a score of zero.
    """
    excerpts = [_normalize(p.full_text_excerpt) for p in state.papers if p.full_text_excerpt]
    gaps_q = [g for g in state.gaps if g.evidence_quotes]
    if not gaps_q or not excerpts:
        return None
    grounded = sum(
        1
        for g in gaps_q
        if any(q and _normalize(q) in e for q in g.evidence_quotes for e in excerpts)
    )
    return grounded / len(gaps_q)


_INLINE_CITATION = re.compile(r"\[(\d+)\]")
_REFERENCES_HEADING = re.compile(r"^#{1,6}\s*references\s*$", re.IGNORECASE | re.MULTILINE)


def citation_coverage(state: ResearchState) -> float | None:
    """Fraction of inline [n] citations in the report body that resolve to a
    numbered reference. None when the report has no inline citation to resolve.
    """
    md = state.report_markdown or ""
    m = _REFERENCES_HEADING.search(md)
    body = md[: m.start()] if m else md
    cited = {int(n) for n in _INLINE_CITATION.findall(body)}
    if not cited:
        return None  # nothing to resolve -- not the same as perfect resolution
    n_refs = len(state.papers)
    resolved = sum(1 for n in cited if 1 <= n <= n_refs)
    return resolved / len(cited)


def gap_keyword_recall(state: ResearchState, case: GoldenCase) -> float | None:
    """Fraction of expected gap keywords occurring as a substring anywhere in the
    gap descriptions. None when the case carries no labels.

    Lexical, not semantic: it credits a keyword that appears in any context,
    including one where the agent asserts the opposite. Read it as a coarse
    topic-overlap signal, not as correctness.
    """
    if not case.expected_gap_keywords:
        return None
    text = _normalize(" ".join(g.description for g in state.gaps))
    hit = sum(1 for kw in case.expected_gap_keywords if _normalize(kw) in text)
    return hit / len(case.expected_gap_keywords)


def paper_recall(state: ResearchState, case: GoldenCase) -> float | None:
    """Fraction of expected papers found, matching an expected title fragment as
    a substring of a retrieved title. None when the case carries no labels.
    """
    if not case.expected_papers:
        return None
    titles = _normalize(" || ".join(p.title for p in state.papers))
    hit = sum(1 for t in case.expected_papers if _normalize(t) in titles)
    return hit / len(case.expected_papers)


def _round(value: float | None, places: int = 2) -> float | None:
    return None if value is None else round(value, places)


def score_case(state: ResearchState, case: GoldenCase) -> dict:
    """All metrics for one finished run against its golden case.

    A None value means "not measurable on this run" and must be excluded from
    aggregates rather than counted as zero or as a perfect score.
    """
    return {
        **operational(state),
        "novelty": state.novelty_score,
        "grounded_in_fulltext": _round(grounded_in_fulltext(state)),
        "citation_coverage": _round(citation_coverage(state)),
        "gap_keyword_recall": _round(gap_keyword_recall(state, case)),
        "paper_recall": _round(paper_recall(state, case)),
        "meets_min_gaps": len(state.gaps) >= case.min_gaps,
    }

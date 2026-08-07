from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from paper_research_agent.core.state import ResearchState
from paper_research_agent.eval import metrics
from paper_research_agent.eval.golden import GoldenCase

# Where a run went wrong, so the report can say *why* the agent fails rather
# than only how often. One run can carry several.
Category = Literal[
    "retrieval_miss",  # expected papers never entered the candidate pool
    "reading_failure",  # full text could not be fetched, chunked or indexed
    "ungrounded_gap",  # gap asserted with no quote, or a quote in no excerpt
    "citation_resolution",  # report cites [n] that resolves to no reference
    "thin_output",  # fewer gaps than the case requires
    "missed_conflict",  # no conflict found on a topic known to be contested
    "provider_error",  # ArXiv/OpenAlex/LLM call failed and was recorded
    "timeout",  # a call or the run hit its time budget
    "query_repetition",  # loop spent a round re-issuing an earlier query
    "novelty_missing",  # an idea was given but never scored
]

# Substrings in state.errors that identify a failure whose cause is visible in
# the message itself. Order matters: first match wins.
_ERROR_SIGNATURES: tuple[tuple[str, Category], ...] = (
    ("timeout", "timeout"),
    ("timed out", "timeout"),
    ("read failed", "reading_failure"),
    ("index failed", "reading_failure"),
    ("search failed", "reading_failure"),
)


class Finding(BaseModel):
    category: Category
    detail: str

    def __str__(self) -> str:  # keeps console output readable
        return f"{self.category}: {self.detail}"


def _classify_error(message: str) -> Category:
    low = message.lower()
    for needle, category in _ERROR_SIGNATURES:
        if needle in low:
            return category
    return "provider_error"


def analyze(state: ResearchState, case: GoldenCase) -> list[Finding]:
    """Categorised failure findings for one finished run. Empty list = clean."""
    findings: list[Finding] = []

    if len(state.gaps) < case.min_gaps:
        findings.append(
            Finding(
                category="thin_output",
                detail=f"{len(state.gaps)} gap(s) < required {case.min_gaps}",
            )
        )

    unquoted = [g for g in state.gaps if not g.evidence_quotes]
    if unquoted:
        findings.append(
            Finding(
                category="ungrounded_gap",
                detail=f"{len(unquoted)}/{len(state.gaps)} gap(s) carry no quote",
            )
        )

    grounding = metrics.grounded_in_fulltext(state)
    if grounding is not None and grounding < 0.5:
        findings.append(
            Finding(
                category="ungrounded_gap",
                detail=f"only {grounding:.0%} of quoted gaps trace to an excerpt",
            )
        )

    read = [p for p in state.papers if p.full_text_excerpt]
    if state.read_full_text and state.papers and not read:
        findings.append(
            Finding(
                category="reading_failure",
                detail="full-text reading was on but no paper yielded an excerpt",
            )
        )

    if case.expected_papers:
        titles = " || ".join(p.title.lower() for p in state.papers)
        missing = [t for t in case.expected_papers if t.lower() not in titles]
        if missing:
            findings.append(
                Finding(
                    category="retrieval_miss",
                    detail=f"expected paper(s) never retrieved: {missing}",
                )
            )

    coverage = metrics.citation_coverage(state)
    if coverage is not None and coverage < 1.0:
        findings.append(
            Finding(
                category="citation_resolution",
                detail=f"{coverage:.0%} of inline citations resolve to a reference",
            )
        )

    if case.expects_conflict and not state.conflicts:
        findings.append(
            Finding(
                category="missed_conflict",
                detail="no conflict found on a topic where the literature disagrees",
            )
        )

    for message in state.errors:
        findings.append(Finding(category=_classify_error(message), detail=message))

    seen: set[str] = set()
    repeated: list[str] = []
    for log in state.round_logs:
        for query in log.queries:
            if query in seen:
                repeated.append(query)
            seen.add(query)
    if repeated:
        findings.append(
            Finding(
                category="query_repetition",
                detail=f"re-issued across rounds: {repeated[:3]}",
            )
        )

    if case.idea and state.novelty_score is None:
        findings.append(
            Finding(category="novelty_missing", detail="idea given but never scored")
        )

    return findings


def distribution(rows: list[dict]) -> dict[str, int]:
    """Count of findings per category across runs, most common first.

    Counts findings, not runs: one run failing three ways contributes to three
    categories. `runs_affected` below is the per-run view.
    """
    counts: dict[str, int] = {}
    for row in rows:
        for finding in row.get("findings", []):
            category = finding["category"] if isinstance(finding, dict) else finding.category
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def runs_affected(rows: list[dict]) -> dict[str, int]:
    "How many distinct runs hit each category at least once."
    counts: dict[str, int] = {}
    for row in rows:
        categories = {
            f["category"] if isinstance(f, dict) else f.category
            for f in row.get("findings", [])
        }
        for category in categories:
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

from __future__ import annotations

from paper_research_agent.core.state import ResearchState
from paper_research_agent.eval import metrics
from paper_research_agent.eval.golden import GoldenCase


def analyze(state: ResearchState, case: GoldenCase) -> list[str]:
    """Heuristic failure findings for one finished run. Empty list = clean."""
    findings: list[str] = []
    gap_text = " ".join(g.description.lower() for g in state.gaps)

    if len(state.gaps) < case.min_gaps:
        findings.append(f"few gaps: {len(state.gaps)} < expected {case.min_gaps}")

    ungrounded = [g for g in state.gaps if not g.evidence_quotes]
    if ungrounded:
        findings.append(f"{len(ungrounded)}/{len(state.gaps)} gap(s) have no evidence quote")

    if state.gaps:
        g = metrics.grounded_in_fulltext(state)
        if g is not None and g < 0.5:
            findings.append(
                f"low full-text grounding ({g:.0%} of quoted gaps trace to an excerpt)"
            )

    if case.expected_gap_keywords:
        missing = [k for k in case.expected_gap_keywords if k.lower() not in gap_text]
        if len(missing) > len(case.expected_gap_keywords) / 2:
            findings.append(f"missed expected gap themes: {missing}")

    if case.expected_papers:
        titles = " || ".join(p.title.lower() for p in state.papers)
        missing = [t for t in case.expected_papers if t.lower() not in titles]
        if missing:
            findings.append(f"expected papers not found: {missing}")

    if not state.conflicts:
        findings.append("no conflicts (ok if undisputed; suspicious on debated topics)")

    if state.errors:
        findings.append(f"provider errors: {state.errors}")

    seen: set[str] = set()
    repeated: list[str] = []
    for log in state.round_logs:
        for query in log.queries:
            if query in seen:
                repeated.append(query)
            seen.add(query)
    if repeated:
        findings.append(f"repeated queries across rounds: {repeated[:3]}")

    if case.idea and state.novelty_score is None:
        findings.append("novelty not scored despite an idea")

    return findings

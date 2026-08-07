from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import Conflict, ResearchGap, ResearchState
from paper_research_agent.eval.failure import analyze, distribution, runs_affected
from paper_research_agent.eval.golden import GoldenCase


def _paper(title: str, *, excerpt: str | None = None) -> Paper:
    return Paper(title=title, authors=["A"], abstract="a", source="arxiv", full_text_excerpt=excerpt)


def _state(**kw) -> ResearchState:
    return ResearchState(topic="t", **kw)


def _categories(state, case) -> set[str]:
    return {f.category for f in analyze(state, case)}


def test_clean_run_produces_no_findings():
    state = _state(
        papers=[_paper("LoRA", excerpt="[Results] adapters cut memory use")],
        gaps=[ResearchGap(description="g", evidence_quotes=["adapters cut memory use"])],
        report_markdown="## Body\nClaim [1].\n\n## References\n[1] LoRA",
    )
    case = GoldenCase(topic="t", min_gaps=1, expected_papers=["lora"])
    assert analyze(state, case) == []


def test_missing_expected_paper_is_a_retrieval_miss():
    state = _state(papers=[_paper("Something else")], gaps=[ResearchGap(description="g", evidence_quotes=["q"])])
    case = GoldenCase(topic="t", min_gaps=1, expected_papers=["lora"])
    assert "retrieval_miss" in _categories(state, case)


def test_no_excerpt_when_reading_was_on_is_a_reading_failure():
    state = _state(read_full_text=True, papers=[_paper("P")], gaps=[ResearchGap(description="g", evidence_quotes=["q"])])
    assert "reading_failure" in _categories(state, GoldenCase(topic="t", min_gaps=1))


def test_gap_without_quote_is_ungrounded():
    state = _state(papers=[_paper("P")], gaps=[ResearchGap(description="g")])
    assert "ungrounded_gap" in _categories(state, GoldenCase(topic="t", min_gaps=1))


def test_too_few_gaps_is_thin_output():
    state = _state(papers=[_paper("P")], gaps=[])
    assert "thin_output" in _categories(state, GoldenCase(topic="t", min_gaps=2))


def test_unresolvable_citation_is_flagged():
    state = _state(
        papers=[_paper("P")],
        gaps=[ResearchGap(description="g", evidence_quotes=["q"])],
        report_markdown="## Body\nClaim [4].\n\n## References\n[1] P",
    )
    assert "citation_resolution" in _categories(state, GoldenCase(topic="t", min_gaps=1))


def test_missed_conflict_only_on_contested_topics():
    state = _state(papers=[_paper("P")], gaps=[ResearchGap(description="g", evidence_quotes=["q"])])
    assert "missed_conflict" in _categories(state, GoldenCase(topic="t", min_gaps=1, expects_conflict=True))
    assert "missed_conflict" not in _categories(state, GoldenCase(topic="t", min_gaps=1))


def test_conflict_found_on_contested_topic_is_clean():
    state = _state(
        papers=[_paper("P")],
        gaps=[ResearchGap(description="g", evidence_quotes=["q"])],
        conflicts=[Conflict(topic="x", position_a="a", position_b="b")],
    )
    assert "missed_conflict" not in _categories(state, GoldenCase(topic="t", min_gaps=1, expects_conflict=True))


def test_timeout_is_separated_from_generic_provider_error():
    state = _state(
        papers=[_paper("P")],
        gaps=[ResearchGap(description="g", evidence_quotes=["q"])],
        errors=["contrast failed: Request timed out", "openalex failed for query 'x': 429"],
    )
    cats = _categories(state, GoldenCase(topic="t", min_gaps=1))
    assert "timeout" in cats
    assert "provider_error" in cats


def test_read_errors_classify_as_reading_failure():
    state = _state(
        papers=[_paper("P", excerpt="e")],
        gaps=[ResearchGap(description="g", evidence_quotes=["e"])],
        errors=["read failed (Some Paper): 404"],
    )
    assert "reading_failure" in _categories(state, GoldenCase(topic="t", min_gaps=1))


def test_novelty_missing_when_an_idea_was_given():
    state = _state(papers=[_paper("P")], gaps=[ResearchGap(description="g", evidence_quotes=["q"])])
    case = GoldenCase(topic="t", idea="my idea", min_gaps=1)
    assert "novelty_missing" in _categories(state, case)


# --- distribution --------------------------------------------------------------


def _rows() -> list[dict]:
    return [
        {"findings": [{"category": "retrieval_miss", "detail": "d"}, {"category": "ungrounded_gap", "detail": "d"}]},
        {"findings": [{"category": "retrieval_miss", "detail": "d"}, {"category": "retrieval_miss", "detail": "d2"}]},
        {"findings": []},
    ]


def test_distribution_counts_findings_sorted_by_frequency():
    dist = distribution(_rows())
    assert list(dist.items())[0] == ("retrieval_miss", 3)
    assert dist["ungrounded_gap"] == 1


def test_runs_affected_counts_each_run_once_per_category():
    affected = runs_affected(_rows())
    assert affected["retrieval_miss"] == 2  # 3 findings, but only 2 runs
    assert affected["ungrounded_gap"] == 1

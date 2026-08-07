from paper_research_agent.core.state import ResearchGap, ResearchState
from paper_research_agent.eval.failure import analyze
from paper_research_agent.eval.golden import GOLDEN, GoldenCase


def test_golden_set_size():
    assert len(GOLDEN) == 13


def test_every_case_carries_provenance():
    "Ground truth whose origin is not recorded is not auditable ground truth."
    for case in GOLDEN:
        assert case.provenance is not None, case.topic
        assert case.provenance.survey_title
        assert case.provenance.arxiv_id
        assert case.provenance.gap_source


def test_every_case_has_labels_to_score():
    for case in GOLDEN:
        assert case.expected_gap_keywords, case.topic
        assert case.expected_papers, case.topic


def test_topics_are_unique():
    topics = [c.topic for c in GOLDEN]
    assert len(set(topics)) == len(topics)


def test_set_covers_a_difficulty_spread():
    assert {c.difficulty for c in GOLDEN} == {"narrow", "broad"}


def test_set_has_contested_topics_to_exercise_conflict_detection():
    contested = [c for c in GOLDEN if c.expects_conflict]
    assert len(contested) >= 2
    # a contested case is grounded in the disagreement itself, not a survey
    for case in contested:
        assert case.provenance.granularity == "debate", case.topic


def test_provenance_granularity_is_classified():
    granularities = {c.provenance.granularity for c in GOLDEN}
    assert granularities <= {"section", "abstract", "debate"}
    # at least some labels come from a survey's own enumerated future-work
    assert sum(c.provenance.granularity == "section" for c in GOLDEN) >= 4


def test_weaker_labels_say_so_in_the_note():
    "Abstract-derived keywords must be flagged, so the README can report them."
    for case in GOLDEN:
        if case.provenance.granularity == "abstract":
            assert "WEAKER GROUND TRUTH" in case.provenance.gap_note, case.topic


# --- failure analysis ----------------------------------------------------------


def _state(**kw) -> ResearchState:
    return ResearchState(topic="t", **kw)


def test_missing_conflict_is_a_finding_only_on_contested_topics():
    state = _state(gaps=[ResearchGap(description="g", evidence_quotes=["q"])])

    settled = GoldenCase(topic="t", min_gaps=1, expects_conflict=False)
    contested = GoldenCase(topic="t", min_gaps=1, expects_conflict=True)

    assert not any("conflict" in f for f in analyze(state, settled))
    assert any("conflict" in f for f in analyze(state, contested))

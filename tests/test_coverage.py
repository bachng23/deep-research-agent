import time

from paper_research_agent.core.state import ResearchGap, ResearchState
from paper_research_agent.features.coverage import assess_coverage, should_continue


def _state_with_gaps(confidences, **kwargs) -> ResearchState:
    state = ResearchState(topic="t", **kwargs)
    state.gaps = [
        ResearchGap(description=f"gap-{i}", confidence=c)
        for i, c in enumerate(confidences)
    ]
    return state


def test_assess_marks_non_high_gaps_as_open():
    state = _state_with_gaps(["high", "medium", "low"])

    assess_coverage(state)

    # only the high-confidence gap is considered closed
    assert state.open_gaps == ["gap-1", "gap-2"]


def test_assess_increments_iteration_and_logs_round():
    state = _state_with_gaps(["medium"], papers=[])

    assess_coverage(state)

    assert state.iteration == 1
    assert len(state.round_logs) == 1
    log = state.round_logs[0]
    assert log.iteration == 1
    assert log.open_gaps == 1
    assert log.closed_gaps == 0


def test_assess_new_papers_is_delta_from_previous_round():
    state = _state_with_gaps(["medium"])

    # round 1: 2 papers total
    state.papers = ["p1", "p2"]  # type: ignore[list-item]
    assess_coverage(state)
    assert state.round_logs[-1].new_papers == 2
    assert state.round_logs[-1].total_papers == 2

    # round 2: grew to 5 total -> 3 new this round
    state.papers = ["p1", "p2", "p3", "p4", "p5"]  # type: ignore[list-item]
    assess_coverage(state)
    assert state.round_logs[-1].new_papers == 3
    assert state.round_logs[-1].total_papers == 5


def test_should_continue_when_open_gaps_and_under_ceiling():
    state = ResearchState(topic="t", max_iterations=3)
    state.open_gaps = ["gap-1"]
    state.iteration = 1

    assert should_continue(state) is True


def test_should_stop_at_ceiling_even_with_open_gaps():
    state = ResearchState(topic="t", max_iterations=3)
    state.open_gaps = ["gap-1"]
    state.iteration = 3

    assert should_continue(state) is False


def test_should_stop_when_no_open_gaps():
    state = ResearchState(topic="t", max_iterations=3)
    state.open_gaps = []
    state.iteration = 1

    assert should_continue(state) is False


def test_should_stop_when_timeout_exceeded():
    state = ResearchState(topic="t", max_iterations=3, timeout_seconds=10.0)
    state.open_gaps = ["gap-1"]
    state.iteration = 1
    state.started_at = time.monotonic() - 11  # 11s elapsed > 10s budget

    assert should_continue(state) is False


def test_timeout_disabled_by_default():
    state = ResearchState(topic="t", max_iterations=3)  # timeout_seconds is None
    state.open_gaps = ["gap-1"]
    state.iteration = 1
    state.started_at = time.monotonic() - 9999  # would trip if timeout were set

    assert should_continue(state) is True

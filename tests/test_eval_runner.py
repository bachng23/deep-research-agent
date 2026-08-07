import json

import pytest

from paper_research_agent.eval import runner


def _row(**kw) -> dict:
    base = {"topic": "t", "difficulty": "narrow", "gap_source": "section", "findings": []}
    base.update(kw)
    return base


def test_aggregate_reports_sample_std_and_n():
    rows = [_row(gaps=2), _row(gaps=4), _row(gaps=6)]
    stat = runner.aggregate(rows)["gaps"]
    assert stat["mean"] == 4.0
    assert stat["std"] == 2.0  # sample std of 2,4,6
    assert stat["n"] == 3


def test_aggregate_drops_none_instead_of_counting_it_as_zero():
    rows = [_row(paper_recall=1.0), _row(paper_recall=None), _row(paper_recall=0.0)]
    stat = runner.aggregate(rows)["paper_recall"]
    assert stat["mean"] == 0.5
    assert stat["n"] == 2  # the None run is excluded, not averaged in as 0


def test_aggregate_has_no_std_for_a_single_point():
    stat = runner.aggregate([_row(gaps=3)])["gaps"]
    assert stat["mean"] == 3.0
    assert stat["std"] is None
    assert stat["n"] == 1


def test_aggregate_handles_a_metric_with_no_data():
    stat = runner.aggregate([_row()])["gaps"]
    assert stat == {"mean": None, "std": None, "n": 0}


def test_crashed_runs_are_excluded_from_metrics_but_counted():
    rows = [_row(gaps=2), {"topic": "t", "run": 1, "crashed": "boom", "findings": []}]
    summary = runner.summarise(rows)
    assert summary["runs_total"] == 2
    assert summary["runs_crashed"] == 1
    assert summary["overall"]["gaps"]["n"] == 1


def test_summary_breaks_down_by_ground_truth_strength():
    rows = [
        _row(gap_source="section", gap_keyword_recall=0.8),
        _row(gap_source="abstract", gap_keyword_recall=0.2),
    ]
    summary = runner.summarise(rows)
    assert summary["by_gap_source"]["section"]["gap_keyword_recall"]["mean"] == 0.8
    assert summary["by_gap_source"]["abstract"]["gap_keyword_recall"]["mean"] == 0.2


def test_per_topic_variance_groups_repeats():
    rows = [_row(topic="a", gaps=1), _row(topic="a", gaps=3), _row(topic="b", gaps=5)]
    per_topic = runner.per_topic_variance(rows)
    assert per_topic["a"]["gaps"]["mean"] == 2.0
    assert per_topic["a"]["gaps"]["std"] == pytest.approx(1.414, abs=0.01)
    assert per_topic["b"]["gaps"]["std"] is None


def test_metadata_records_what_produced_the_numbers(monkeypatch):
    monkeypatch.setenv("API_KEY", "x")
    meta = runner.run_metadata(repeats=3, max_iterations=2, n_cases=13)

    assert meta["llm"]["temperature"] == 0.0
    assert meta["llm"]["seed"] is None  # recorded as unfixed, not silently omitted
    assert meta["eval"] == {
        "cases": 13,
        "repeats": 3,
        "max_iterations": 2,
        "read_full_text": True,
        "use_memory": False,
    }
    for tier in ("fast", "balanced", "reasoning", "embedding"):
        assert meta["models"][tier]
    assert meta["timestamp_utc"].endswith("+00:00")
    # git fields may be None outside a checkout, but the keys must exist
    assert "git_sha" in meta and "git_dirty" in meta


def test_metadata_is_json_serialisable():
    json.dumps(runner.run_metadata(repeats=1, max_iterations=1, n_cases=1))


def test_slug_is_filesystem_safe():
    assert runner._slug("RAG vs. long-context LMs!") == "rag-vs-long-context-lms"

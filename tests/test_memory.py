import string

import pytest

import paper_research_agent.features.reading.node as rnode
import paper_research_agent.memory.nodes as mnodes
import paper_research_agent.memory.results as results_mod
from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import ResearchGap, ResearchState
from paper_research_agent.features.reading import read_papers
from paper_research_agent.memory.nodes import recall_prior, remember_result
from paper_research_agent.memory.results import ResultMemory


@pytest.fixture(autouse=True)
def _fake_topic_embeddings(monkeypatch):
    "Offline topic embeddings: char-presence vector (same topic -> identical)."

    class _FakeEmb:
        def embed_query(self, text: str):
            t = text.lower()
            return [1.0 if c in t else 0.0 for c in string.ascii_lowercase]

    monkeypatch.setattr(results_mod, "embeddings_model", lambda: _FakeEmb())
    monkeypatch.setattr(results_mod, "_normalize_topic", lambda topic: topic)


def _result_state(topic, gap_descriptions):
    return ResearchState(
        topic=topic, gaps=[ResearchGap(description=d) for d in gap_descriptions]
    )


class _FakeMemory:
    """Stand-in for PaperMemory: knows() is forced; records remember() calls."""

    def __init__(self, knows_all: bool = False):
        self._knows_all = knows_all
        self.remembered: list[str] = []

    def knows(self, paper_id: str) -> bool:
        return self._knows_all

    def remember(self, paper_id: str, chunks) -> None:
        self.remembered.append(paper_id)

    def relevant(self, focus, *, paper_id=None, top_k=5):
        return [{"section": "Results", "text": "cached finding"}]

    def close(self):
        pass


def _state():
    return ResearchState(
        topic="t",
        papers=[Paper(title="P1", source="arxiv", url="http://p1")],
        read_full_text=True,
    )


def test_memory_hit_skips_fetch(monkeypatch):
    # a remembered paper must NOT be fetched/embedded again
    def _boom(paper):
        raise AssertionError("should not fetch a remembered paper")

    fake = _FakeMemory(knows_all=True)
    monkeypatch.setattr(rnode, "PaperMemory", lambda path: fake)
    monkeypatch.setattr(rnode, "fetch_full_text", _boom)

    out = read_papers(_state())

    assert out.papers[0].full_text_excerpt is not None  # built from cached chunks
    assert fake.remembered == []  # nothing re-indexed


def test_memory_miss_reads_and_remembers(monkeypatch):
    fake = _FakeMemory(knows_all=False)
    monkeypatch.setattr(rnode, "PaperMemory", lambda path: fake)
    monkeypatch.setattr(rnode, "fetch_full_text", lambda p: "## Results\ncats are great")

    out = read_papers(_state())

    assert fake.remembered == ["http://p1"]  # new paper got indexed
    assert out.papers[0].full_text_excerpt is not None


def test_memory_gate_off_is_noop(monkeypatch):
    monkeypatch.setattr(
        rnode, "PaperMemory",
        lambda path: (_ for _ in ()).throw(AssertionError("memory built while gated off")),
    )
    state = ResearchState(
        topic="t",
        papers=[Paper(title="P1", source="arxiv", url="http://p1")],
        read_full_text=False,
    )
    out = read_papers(state)
    assert out.papers[0].full_text_excerpt is None


# ---------------- episodic memory (ResultMemory + nodes) ----------------

def test_result_memory_recall_related(tmp_path):
    mem = ResultMemory(str(tmp_path / "results.db"))
    mem.remember(_result_state(
        "hierarchical chunking retrieval augmented generation",
        ["no table evaluation", "missing long document benchmark"],
    ))
    recalled = mem.recall("retrieval augmented generation over scientific documents")
    assert "no table evaluation" in recalled  # shared keywords -> prior gaps surface


def test_result_memory_recall_unrelated_is_empty(tmp_path):
    mem = ResultMemory(str(tmp_path / "results.db"))
    mem.remember(_result_state("quantum error correction codes", ["g1"]))
    assert mem.recall("transformer attention mechanisms") == []


def test_result_memory_noop_without_path():
    mem = ResultMemory(None)
    mem.remember(_result_state("t", ["g"]))  # no-op, no error
    assert mem.recall("t") == []


def test_nodes_gate_off(monkeypatch):
    monkeypatch.setattr(mnodes, "_result_path", lambda state: None)
    out = recall_prior(ResearchState(topic="t"))
    assert out.recalled_gaps == []
    remember_result(_result_state("t", ["g"]))  # no-op, no error


def test_nodes_remember_then_recall(monkeypatch, tmp_path):
    monkeypatch.setattr(mnodes, "_result_path", lambda state: str(tmp_path / "results.db"))
    remember_result(_result_state("retrieval augmented generation chunking", ["table eval gap"]))
    out = recall_prior(ResearchState(topic="retrieval augmented generation documents"))
    assert "table eval gap" in out.recalled_gaps


# ---------------- TTL result-cache (fresh + short-circuit) ----------------

def test_fresh_within_and_outside_ttl(tmp_path):
    mem = ResultMemory(str(tmp_path / "results.db"))
    mem.remember(_result_state("topic A", ["g1"]))

    got = mem.fresh("topic A", 7)
    assert got is not None and got.topic == "topic A"
    assert [g.description for g in got.gaps] == ["g1"]
    assert mem.fresh("topic A", 0) is None      # ttl 0 -> never reuse
    assert mem.fresh("zzz qqq", 7) is None       # dissimilar topic


def test_fresh_fuzzy_topic_match(tmp_path):
    mem = ResultMemory(str(tmp_path / "results.db"))
    mem.remember(_result_state("retrieval augmented generation", ["g"]))
    # lexically different phrasing of the same topic still hits (semantic-ish)
    assert mem.fresh("retrieval augmented generation systems", 7) is not None


def test_fresh_expired(tmp_path):
    import time

    mem = ResultMemory(str(tmp_path / "results.db"))
    mem.remember(_result_state("old topic", ["g"]))
    mem._conn.execute("UPDATE runs SET created = ?", (time.time() - 10 * 86400,))
    mem._conn.commit()
    assert mem.fresh("old topic", 7) is None     # researched 10d ago, TTL 7d


def test_run_research_short_circuits_on_cache(monkeypatch):
    import paper_research_agent.agent.graph as graph_mod

    cached = ResearchState(topic="x", report_markdown="cached")
    monkeypatch.setattr(graph_mod, "_cached_result", lambda topic, use_memory: cached)
    monkeypatch.setattr(
        graph_mod, "build_graph",
        lambda: (_ for _ in ()).throw(AssertionError("graph built despite cache")),
    )
    out = graph_mod.run_research("x", use_memory=True)
    assert out is cached

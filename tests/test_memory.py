import paper_research_agent.features.reading.node as rnode
import paper_research_agent.memory.nodes as mnodes
from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import ResearchGap, ResearchState
from paper_research_agent.features.reading import read_papers
from paper_research_agent.memory.nodes import recall_prior, remember_result
from paper_research_agent.memory.results import ResultMemory


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

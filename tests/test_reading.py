import paper_research_agent.features.reading.fetcher as fetcher
import paper_research_agent.features.reading.indexing as indexing
import paper_research_agent.features.reading.node as rnode
from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import ResearchGap, ResearchState
from paper_research_agent.features.reading.chunking import Chunk, chunk_text
from paper_research_agent.features.reading.excerpt import build_excerpt
from paper_research_agent.features.reading.fetcher import (
    _arxiv_id_from,
    _parse_arxiv_html,
    fetch_full_text,
)
from paper_research_agent.features.reading.indexing import FullTextIndexing
from paper_research_agent.features.reading.node import _focus, _select_papers, read_papers


# ---------------- chunking ----------------

def test_chunk_splits_by_markdown_headings():
    chunks = chunk_text("## Intro\nhello\n## Results\nwe find X")
    assert {"Intro", "Results"} <= {c.section for c in chunks}


def test_chunk_flags_findings_sections():
    chunks = chunk_text("## Introduction\nfoo\n## Results\nbar")
    flag = {c.section: c.is_findings for c in chunks}
    assert flag["Results"] is True
    assert flag["Introduction"] is False


def test_chunk_windows_long_section_with_overlap():
    body = "".join(str(i % 10) for i in range(3500))
    chunks = [c for c in chunk_text(f"## Big\n{body}", max_chars=1000, overlap=200)
              if c.section == "Big"]
    assert len(chunks) >= 4
    assert chunks[0].text[-200:] == chunks[1].text[:200]  # overlap math


def test_chunk_no_headings_single_section():
    chunks = chunk_text("just plain text, no headings")
    assert len(chunks) == 1
    assert chunks[0].section is None


# ---------------- excerpt ----------------

def test_build_excerpt_labels_sections():
    out = build_excerpt(
        [{"section": "Results", "text": "finding one"},
         {"section": None, "text": "body two"}],
        max_chars=1000,
    )
    assert "[Results] finding one" in out
    assert "[Body] body two" in out


def test_build_excerpt_bounds_by_max_chars():
    out = build_excerpt(
        [{"section": "A", "text": "x" * 100}, {"section": "B", "text": "y" * 100}],
        max_chars=120,
    )
    assert "x" * 100 in out
    assert "y" * 100 not in out


def test_build_excerpt_empty_returns_none():
    assert build_excerpt([], max_chars=1000) is None


# ---------------- fetcher ----------------

def test_arxiv_id_from_various_urls():
    def pid(u):
        return _arxiv_id_from(Paper(title="t", source="arxiv", url=u))

    assert pid("https://arxiv.org/abs/2312.10997") == "2312.10997"
    assert pid("https://arxiv.org/pdf/2312.10997v2.pdf") == "2312.10997"
    assert pid("https://doi.org/10.1/x") is None


def test_parse_arxiv_html_drops_bibliography():
    html = """
    <div class="ltx_page_content">
      <h2 class="ltx_title ltx_title_section">1 Intro</h2>
      <p class="ltx_p">hello world</p>
      <section class="ltx_bibliography">
        <h2 class="ltx_title ltx_title_bibliography">References</h2>
        <p class="ltx_p">[1] Smith et al arXiv preprint</p>
      </section>
    </div>"""
    md = _parse_arxiv_html(html)
    assert "## 1 Intro" in md
    assert "hello world" in md
    assert "References" not in md
    assert "Smith" not in md


def test_fetch_full_text_falls_back_html_to_pdf(monkeypatch):
    monkeypatch.setattr(fetcher, "_from_arxiv_html", lambda aid: None)
    monkeypatch.setattr(fetcher, "_from_pdf", lambda url: "PDF TEXT")
    p = Paper(title="t", source="arxiv",
              url="https://arxiv.org/abs/2312.10997",
              pdf_url="https://arxiv.org/pdf/2312.10997")
    assert fetch_full_text(p) == "PDF TEXT"


def test_fetch_full_text_none_when_no_sources():
    p = Paper(title="t", source="openalex", url="https://doi.org/x", pdf_url=None)
    assert fetch_full_text(p) is None


# ---------------- indexing (mock embeddings, real in-memory Qdrant) ----------------

class _FakeEmb:
    """Deterministic 2-D vectors: 'cat' -> [1,0], else [0,1]."""

    def _vec(self, text):
        return [1.0, 0.0] if "cat" in text else [0.0, 1.0]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_index_search_cross_paper_and_filter(monkeypatch):
    monkeypatch.setattr(indexing, "embeddings_model", lambda: _FakeEmb())
    idx = FullTextIndexing()
    idx.add("p1", [Chunk(text="cat sits", section="A")])
    idx.add("p2", [Chunk(text="dog runs", section="B")])

    assert idx.search("cat", top_k=5)[0]["paper_id"] == "p1"          # cross-paper
    assert all(h["paper_id"] == "p2"
               for h in idx.search("cat", paper_id="p2", top_k=5))     # filter


def test_index_search_empty_before_add(monkeypatch):
    monkeypatch.setattr(indexing, "embeddings_model", lambda: _FakeEmb())
    assert FullTextIndexing().search("x") == []


def test_index_persists_across_instances(monkeypatch, tmp_path):
    "On-disk index survives a new FullTextIndexing (i.e. a new run/process)."
    monkeypatch.setattr(indexing, "embeddings_model", lambda: _FakeEmb())
    path = str(tmp_path / "qdrant")

    idx = FullTextIndexing(path=path)
    idx.add("p1", [Chunk(text="cat sits", section="A")])
    idx._client.close()  # release the on-disk lock (simulate the run ending)

    reopened = FullTextIndexing(path=path)
    assert reopened.has_paper("p1")  # persisted across instances
    assert reopened.search("cat", paper_id="p1")
    reopened._client.close()


def test_index_idempotent_on_reindex(monkeypatch, tmp_path):
    "Re-indexing the same paper overwrites (deterministic ids) -> no duplicates."
    monkeypatch.setattr(indexing, "embeddings_model", lambda: _FakeEmb())
    path = str(tmp_path / "qdrant")

    idx = FullTextIndexing(path=path)
    idx.add("p1", [Chunk(text="cat sits", section="A")])
    idx.add("p1", [Chunk(text="cat sits", section="A")])  # same chunk again
    hits = idx.search("cat", paper_id="p1", top_k=10)
    assert len(hits) == 1
    idx._client.close()


# ---------------- node ----------------

def test_read_papers_gate_off_is_noop():
    p = Paper(title="P", source="arxiv", url="u")
    out = read_papers(ResearchState(topic="t", papers=[p], read_full_text=False))
    assert out.papers[0].full_text_excerpt is None


def test_read_papers_populates_excerpt(monkeypatch):
    monkeypatch.setattr(rnode, "fetch_full_text",
                        lambda p: "## Results\ncats are great")

    class _Mem:
        def __init__(self, path=None):
            pass

        def knows(self, pid):
            return False

        def remember(self, pid, chunks):
            pass

        def relevant(self, focus, paper_id=None, top_k=5):
            return [{"section": "Results", "text": "cats are great"}]

        def close(self):
            pass

    monkeypatch.setattr(rnode, "PaperMemory", _Mem)
    p = Paper(title="P", source="arxiv", url="u")
    out = read_papers(ResearchState(topic="t", papers=[p], read_full_text=True))
    assert "cats are great" in out.papers[0].full_text_excerpt


def test_select_papers_prioritizes_gap_cited():
    p1 = Paper(title="A", source="arxiv", url="1")
    p2 = Paper(title="B", source="arxiv", url="2")
    state = ResearchState(
        topic="t", papers=[p1, p2],
        gaps=[ResearchGap(description="g", supporting_papers=["B"])],
    )
    assert _select_papers(state, top_k=2)[0].title == "B"


def test_focus_includes_topic_idea_gaps():
    f = _focus(ResearchState(topic="T", user_idea="I",
                             gaps=[ResearchGap(description="G")]))
    assert "T" in f and "I" in f and "G" in f

from __future__ import annotations

from paper_research_agent.config import get_settings
from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import ResearchState
from paper_research_agent.features.fetching.dedup import paper_id
from paper_research_agent.features.reading import build_excerpt, fetch_full_text
from paper_research_agent.features.reading.chunking import chunk_text
from paper_research_agent.memory import PaperMemory


def read_papers(state: ResearchState) -> ResearchState:
    """
    Read full text of the top-K papers and store focus-relevant excerpts.
    Finish-path node: abstracts already mapped the territory; here we read deeply so conflicts/novelty (and the gap-refine pass) get Results/Dicussion text.
    Gated by 'read_full_text'; degrades to abstract on any failure.
    """

    if not state.read_full_text or not state.papers:
        return state

    settings = get_settings()
    targets = _select_papers(state, settings.read_max_papers)
    if not targets:
        return state

    path = settings.memory_dir if settings.use_memory else None
    memory = PaperMemory(path)

    try:
        indexed: set[str] = set()

        # read + chunk + index selected papers
        for paper in targets:
            pid = paper_id(paper)
            if memory.knows(pid):  # memory hit -> already embedded, reuse
                indexed.add(pid)
                continue
            try:
                text = fetch_full_text(paper)
            except Exception as e:
                state.errors.append(f"read failed ({paper.title[:40]}): {e}")
                continue
            if not text:
                continue

            chunks = chunk_text(
                text,
                max_chars=settings.fulltext_chunk_chars,
                overlap=settings.fulltext_chunk_overlap,
            )
            if not chunks:
                continue

            try:
                memory.remember(pid, chunks)
                indexed.add(pid)
            except Exception as e:
                state.errors.append(f"index failed ({paper.title[:40]}): {e}")

        if not indexed:
            return state

        # retrieve focus-relevant excerpts per paper
        focus = _focus(state)
        for paper in targets:
            pid = paper_id(paper)
            if pid not in indexed:
                continue
            try:
                hits = memory.relevant(
                    focus, paper_id=pid, top_k=settings.fulltext_top_k
                )
            except Exception as e:
                state.errors.append(f"search failed ({paper.title[:40]}): {e}")
                continue
            paper.full_text_excerpt = build_excerpt(
                hits, max_chars=settings.fulltext_excerpt_max_chars
            )
    finally:
        memory.close()
    return state


def _select_papers(state: ResearchState, top_k: int) -> list[Paper]:
    "Top-K to read: papers cited by gaps first (by gap order), then fill by order."
    by_title = {_norm(p.title): p for p in state.papers}
    ordered: list[Paper] = []
    seen: set[str] = set()

    def _add(paper: Paper | None) -> None:
        if paper is None:
            return
        pid = paper_id(paper)
        if pid not in seen:
            seen.add(pid)
            ordered.append(paper)

    for gap in state.gaps:
        for title in gap.supporting_papers:
            _add(by_title.get(_norm(title)))

    for paper in state.papers:
        _add(paper)

    ordered.sort(key=_readable, reverse=True)

    return ordered[:top_k]


def _readable(paper: Paper) -> bool:
    return "arxiv.org" in (paper.url or "") or bool(paper.pdf_url)


def _focus(state: ResearchState) -> str:
    parts = [state.topic]
    if state.user_idea:
        parts.append(state.user_idea)
    parts.extend(g.description for g in state.gaps)
    return "\n".join(parts)


def _norm(title: str) -> str:
    return " ".join(title.lower().split())

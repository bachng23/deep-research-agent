from __future__ import annotations

from paper_research_agent.features.reading.chunking import Chunk
from paper_research_agent.features.reading.indexing import FullTextIndexing


class PaperMemory:
    """The agent's long-term memory of papers it has read: a persistent
    full-text index keyed by paper_id. path=None -> ephemeral (this run only)."""

    def __init__(self, path: str | None = None) -> None:
        self._index = FullTextIndexing(path=path)

    def knows(self, paper_id: str) -> bool:
        return self._index.has_paper(paper_id)

    def remember(self, paper_id: str, chunks: list[Chunk]) -> None:
        self._index.add(paper_id, chunks)

    def relevant(self, focus: str, *, paper_id: str | None = None, top_k: int = 5):
        return self._index.search(focus, paper_id=paper_id, top_k=top_k)

    def close(self) -> None:
        self._index.close()

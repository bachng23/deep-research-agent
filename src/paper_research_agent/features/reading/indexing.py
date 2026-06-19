from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from paper_research_agent.features.reading.chunking import Chunk
from paper_research_agent.llm import embeddings_model


def _point_id(paper_id: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"paper-research:{paper_id}:{index}"))


class FullTextIndexing:
    """
    Qdrant index of full-text chunks across ALL papers.
    path=None -> in-memory (per run); path=<dir> -> on-disk (persists across runs).
    One collection; every chunk carries 'paper_id' in its payload, so it is possible
    to retrieve per-paper (filter) or cross-paper (no filter) from the same store.
    """

    COLLECTION = "papers"

    def __init__(self, path: str | None = None) -> None:
        self._client = QdrantClient(path=path) if path else QdrantClient(":memory:")
        self._emb = embeddings_model()
        self._ready = self._client.collection_exists(self.COLLECTION)

    def add(self, paper_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self._emb.embed_documents([c.text for c in chunks])

        self._ensure_collection(len(vectors[0]))
        self._client.upsert(
            self.COLLECTION,
            points=[
                models.PointStruct(
                    id=_point_id(paper_id, i),
                    vector=vec,
                    payload={
                        "paper_id": paper_id,
                        "section": c.section,
                        "is_findings": c.is_findings,
                        "text": c.text,
                    },
                )
                for i, (c, vec) in enumerate(zip(chunks, vectors))
            ],
        )

    def has_paper(self, paper_id: str) -> bool:
        "If already indexed, skipping re-reading or re-embedding."
        if not self._ready:
            return False
        points, _ = self._client.scroll(
            self.COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="paper_id", match=models.MatchValue(value=paper_id)
                    )
                ]
            ),
            limit=1,
        )
        return bool(points)

    def search(
        self, focus: str, *, paper_id: str | None = None, top_k: int = 5
    ) -> list[dict]:
        if not self._ready:
            return []

        query_filter = None
        if paper_id is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="paper_id", match=models.MatchValue(value=paper_id)
                    )
                ]
            )

        hits = self._client.query_points(
            self.COLLECTION,
            query=self._emb.embed_query(focus),
            query_filter=query_filter,
            limit=top_k,
        ).points

        return [{**h.payload, "score": h.score} for h in hits]

    def _ensure_collection(self, dim: int) -> None:
        if self._ready:
            return
        self._client.create_collection(
            self.COLLECTION,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            ),
        )
        self._ready = True

    def close(self) -> None:
        self._client.close()

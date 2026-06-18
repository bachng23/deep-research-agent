from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from paper_research_agent.features.reading.chunking import Chunk
from paper_research_agent.llm import embeddings_model


class FullTextIndexing:
    """
    In-memory Qdrant index of full-text chunks across ALL paper.
    One collection; every chunk carries 'paper_id' in its payload, so it is possible to retrieve per-paper (filter) or cross-paper (no filter) from the same store.
    """

    COLLECTION = "papers"

    def __init__(self):
        self._client = QdrantClient(":memory:")
        self._emb = embeddings_model()
        self._ready = False

    def add(self, paper_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self._emb.embed_documents([c.text for c in chunks])

        self._ensure_collection(len(vectors[0]))
        self._client.upsert(
            self.COLLECTION,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "paper_id": paper_id,
                        "section": c.section,
                        "is_findings": c.is_findings,
                        "text": c.text,
                    },
                )
                for c, vec in zip(chunks, vectors)
            ],
        )

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

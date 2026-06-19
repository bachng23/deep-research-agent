from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from paper_research_agent.core.state import ResearchState
from paper_research_agent.features.contrast.ranking import _keywords  # reuse
from paper_research_agent.llm import chat_model_for_tier, embeddings_model

_NORMALIZE_SYSTEM = (
    "Expand acronyms and return ONLY the canonical full research topic name "
    "(no quotes, no explanation). E.g. 'RAG' -> 'Retrieval Augmented Generation'."
)


def _normalize_topic(topic: str) -> str:
    "Canonicalize a topic via a fast LLM so acronyms match their expansions."
    try:
        resp = chat_model_for_tier("fast").invoke(
            [("system", _NORMALIZE_SYSTEM), ("user", topic)]
        )
        return str(resp.content).strip() or topic
    except Exception:
        return topic


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class ResultMemory:
    """Past runs: full result per topic + a topic embedding + timestamp.

    Powers TTL reuse with *semantic* topic matching ("RAG" hits a prior
    "retrieval-augmented generation" run) and related-topic gap recall.
    path=None -> no-op (gated off)."""

    def __init__(self, path: str | None = None) -> None:
        self._conn: sqlite3.Connection | None = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS runs "
                "(topic TEXT PRIMARY KEY, state TEXT, embedding TEXT, created REAL)"
            )
            self._conn.commit()

    def remember(self, state: ResearchState) -> None:
        if self._conn is None or not state.gaps:
            return
        try:
            emb = embeddings_model().embed_query(_normalize_topic(state.topic))
        except Exception:
            emb = []  # embeddings down -> store without it (won't match semantically)
        self._conn.execute(
            "INSERT OR REPLACE INTO runs (topic, state, embedding, created) "
            "VALUES (?, ?, ?, ?)",
            (state.topic, state.model_dump_json(), json.dumps(emb), time.time()),
        )
        self._conn.commit()

    def fresh(
        self, topic: str, ttl_days: int, *, min_similarity: float = 0.75
    ) -> ResearchState | None:
        "Cached result for the most similar fresh topic (>= min_similarity); else None."
        if self._conn is None or ttl_days <= 0:
            return None
        cutoff = time.time() - ttl_days * 86400
        rows = self._conn.execute(
            "SELECT state, embedding FROM runs WHERE created >= ?", (cutoff,)
        ).fetchall()
        if not rows:
            return None

        try:
            query = embeddings_model().embed_query(_normalize_topic(topic))
        except Exception:
            return None

        best_sim, best_state = 0.0, None
        for state_json, emb_json in rows:
            sim = _cosine(query, json.loads(emb_json))
            if sim > best_sim:
                best_sim, best_state = sim, state_json

        if best_state is not None and best_sim >= min_similarity:
            return ResearchState.model_validate_json(best_state)
        return None

    def recall(self, topic: str, *, top_k: int = 3) -> list[str]:
        "Gap descriptions from past runs on keyword-related topics."
        if self._conn is None:
            return []
        kw = _keywords(topic)
        scored: list[tuple[int, list[str]]] = []
        for past_topic, state_json in self._conn.execute(
            "SELECT topic, state FROM runs"
        ):
            if past_topic == topic:
                continue
            overlap = len(kw & _keywords(past_topic))
            if overlap:
                gaps = [g["description"] for g in json.loads(state_json).get("gaps", [])]
                scored.append((overlap, gaps))
        scored.sort(key=lambda x: x[0], reverse=True)
        seen: set[str] = set()
        out: list[str] = []
        for _, gaps in scored[:top_k]:
            for g in gaps:
                if g not in seen:
                    seen.add(g)
                    out.append(g)
        return out

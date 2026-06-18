from __future__ import annotations


def build_excerpt(hits: list[dict], *, max_chars: int = 4000) -> str | None:
    """
    Join retrieved chunks into a bounded, source-labelled excerpt.
    Hits are assumed already sorted by relevance (Qdrant score).
    """
    if not hits:
        return None

    blocks, total = [], 0
    for h in hits:
        label = h.get("section") or "Body"
        block = f"[{label}] {h['text'].strip()}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)

    return "\n\n".join(blocks) or None

from __future__ import annotations

from difflib import SequenceMatcher

from paper_research_agent.core.models import Paper


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    unique: list[Paper] = []

    for paper in papers:
        if not is_duplicate(paper, unique):
            unique.append(paper)

    return unique


def is_duplicate(paper: Paper, existing_papers: list[Paper]) -> bool:
    normalized_title = normalize_title(paper.title)

    for existing in existing_papers:
        existing_title = normalize_title(existing.title)

        if normalized_title == existing_title:
            return True

        similarity = SequenceMatcher(None, normalized_title, existing_title).ratio()
        if similarity >= 0.92:
            return True

    return False


def paper_id(paper: Paper) -> str:
    "Stable identity for cross-round dedup: prefer URL, fall back to title."
    if paper.url:
        return paper.url.strip().lower()
    return normalize_title(paper.title)


def normalize_title(title: str) -> str:
    return " ".join(title.lower().strip().split())

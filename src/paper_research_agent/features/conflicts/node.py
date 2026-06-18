from __future__ import annotations

from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import Conflict, ResearchState
from paper_research_agent.features.conflicts.prompts import (
    CONFLICTS_SYSTEM_PROMPT,
    CONFLICTS_USER_PROMPT,
)
from paper_research_agent.features.conflicts.schemas import ConflictAnalysis
from paper_research_agent.llm import chat_model_for_tier, invoke_with_retry


def find_conflicts(state: ResearchState) -> ResearchState:
    if len(state.papers) < 2:
        return state

    try:
        state.conflicts = _find_conflicts_with_llm(state)
    except Exception as e:
        state.errors.append(f"conflicts failed: {e}")

    return state


def _find_conflicts_with_llm(state: ResearchState) -> list[Conflict]:
    model = chat_model_for_tier("reasoning").with_structured_output(ConflictAnalysis)

    prompt = CONFLICTS_USER_PROMPT.format(
        topic=state.topic,
        papers=_format_papers(state.papers),
    )
    result = invoke_with_retry(
        model, [("system", CONFLICTS_SYSTEM_PROMPT), ("user", prompt)]
    )

    return result.conflicts


def _format_papers(papers: list[Paper]) -> str:
    blocks: list[str] = []
    for i, paper in enumerate(papers, start=1):
        body = (
            paper.full_text_excerpt or paper.abstract or "No text available."
        ).strip()
        blocks.append(f"[{i}] {paper.title} ({paper.year})\n{body}")
    return "\n\n".join(blocks)

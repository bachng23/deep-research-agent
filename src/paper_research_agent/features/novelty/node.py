from __future__ import annotations

from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import ResearchGap, ResearchState
from paper_research_agent.features.novelty.prompts import (
    NOVELTY_SYSTEM_PROMPT,
    NOVELTY_USER_PROMPT,
)
from paper_research_agent.features.novelty.schemas import NoveltyAssessment
from paper_research_agent.llm import chat_model_for_tier, invoke_with_retry


def score_novelty(state: ResearchState) -> ResearchState:
    if not state.user_idea or not state.papers:
        return state

    try:
        assessment = _score_novelty_with_llm(state)
        state.novelty_score = assessment.score
        state.novelty_reasoning = assessment.reasoning
        state.overlapping_papers = assessment.overlapping_papers
    except Exception as e:
        state.errors.append(f"novelty failed: {e}")

    return state


def _score_novelty_with_llm(state: ResearchState) -> NoveltyAssessment:
    model = chat_model_for_tier("reasoning")
    structured_model = model.with_structured_output(NoveltyAssessment)

    prompt = NOVELTY_USER_PROMPT.format(
        topic=state.topic,
        user_idea=state.user_idea,
        gaps=_format_gaps(state.gaps),
        papers=_format_papers(state.papers),
    )

    return invoke_with_retry(
        structured_model,
        [
            ("system", NOVELTY_SYSTEM_PROMPT),
            ("user", prompt),
        ],
    )


def _format_gaps(gaps: list[ResearchGap]) -> str:
    if not gaps:
        return "No gaps identified."

    return "\n".join(
        f"- {gap.description} (confidence: {gap.confidence})" for gap in gaps
    )


def _format_papers(papers: list[Paper]) -> str:
    blocks: list[str] = []

    for i, paper in enumerate(papers, start=1):
        abstract = (paper.abstract or "No abstract available.").strip()
        blocks.append(f"[{i}] {paper.title} ({paper.year})\n{abstract}")

    return "\n\n".join(blocks)

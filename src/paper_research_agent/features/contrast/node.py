from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import ResearchGap, ResearchState
from paper_research_agent.features.contrast.prompts import (
    CONTRAST_SYSTEM_PROMPT,
    CONTRAST_UPDATE_USER_PROMPT,
    CONTRAST_USER_PROMPT,
)
from paper_research_agent.features.contrast.schemas import GapAnalysis
from paper_research_agent.features.fetching.dedup import paper_id
from paper_research_agent.llm import chat_model_for_tier


def find_gaps(state: ResearchState) -> ResearchState:
    if not state.papers:
        return state

    try:
        if state.gaps:
            state.gaps = _update_gaps_with_llm(state)  # round 2+: refine
        else:
            state.gaps = _discover_gaps_with_llm(state)  # round 1: discover
    except Exception as e:
        state.errors.append(f"contrast failed: {e}")

    return state


def _discover_gaps_with_llm(state: ResearchState) -> list[ResearchGap]:
    prompt = CONTRAST_USER_PROMPT.format(
        topic=state.topic,
        user_idea=state.user_idea or "Not provided",
        papers=_format_papers(state.papers),
    )

    result = _structured_model().invoke(
        [
            ("system", CONTRAST_SYSTEM_PROMPT),
            ("user", prompt),
        ]
    )

    return result.gaps


def _update_gaps_with_llm(state: ResearchState) -> list[ResearchGap]:
    new_papers = _recent_papers(state)
    if not new_papers:
        return state.gaps  # nothing new, then keep gaps unchanged

    prompt = CONTRAST_UPDATE_USER_PROMPT.format(
        topic=state.topic,
        gaps=_format_gaps(state.gaps),
        papers=_format_papers(new_papers),
    )

    result = _structured_model().invoke(
        [
            ("system", CONTRAST_SYSTEM_PROMPT),
            ("user", prompt),
        ]
    )

    updated = result.gaps

    return updated or state.gaps


def _structured_model():
    return chat_model_for_tier("reasoning").with_structured_output(GapAnalysis)


def _recent_papers(state: ResearchState) -> list[Paper]:
    recent = set(state.recent_paper_ids)
    return [p for p in state.papers if paper_id(p) in recent]


def _format_gaps(gaps: list[ResearchGap]) -> str:
    return "\n".join(
        f"[{i}] ({gap.confidence}) {gap.description}"
        for i, gap in enumerate(gaps, start=1)
    )


def _format_papers(papers: list[Paper]) -> str:
    blocks: list[str] = []

    for i, paper in enumerate(papers, start=1):
        abstract = (paper.abstract or "No abstract available.").strip()
        blocks.append(f"[{i}] {paper.title} ({paper.year})\n{abstract}")

    return "\n\n".join(blocks)

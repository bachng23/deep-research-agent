from paper_research_agent.core.models import Paper
from paper_research_agent.core.state import ResearchGap, ResearchState
from paper_research_agent.features.contrast.prompts import (
    CONTRAST_SYSTEM_PROMPT,
    CONTRAST_USER_PROMPT,
)
from paper_research_agent.features.contrast.schemas import GapAnalysis
from paper_research_agent.llm import chat_model_for_tier


def find_gaps(state: ResearchState) -> ResearchState:
    if not state.papers:
        return state

    try:
        state.gaps = _find_gaps_with_llm(state)
    except Exception as e:
        state.errors.append(f"contrast failed: {e}")

    return state


def _find_gaps_with_llm(state: ResearchState) -> list[ResearchGap]:
    model = chat_model_for_tier("reasoning")
    structured_model = model.with_structured_output(GapAnalysis)

    user_idea = state.user_idea or "Not Provided"
    papers_block = _format_papers(state.papers)

    prompt = CONTRAST_USER_PROMPT.format(
        topic=state.topic,
        user_idea=user_idea,
        papers=papers_block,
    )

    result = structured_model.invoke(
        [("system", CONTRAST_SYSTEM_PROMPT), ("user", prompt)]
    )

    return result.gaps


def _format_papers(papers: list[Paper]) -> str:
    blocks: list[str] = []

    for i, paper in enumerate(papers, start=1):
        abstract = (paper.abstract or "No abstract available.").strip()
        blocks.append(f"[{i}] {paper.title} ({paper.year})\n{abstract}")

    return "\n\n".join(blocks)

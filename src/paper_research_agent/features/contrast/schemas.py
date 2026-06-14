from pydantic import BaseModel, Field

from paper_research_agent.core.state import ResearchGap


class GapAnalysis(BaseModel):
    gaps: list[ResearchGap] = Field(
        default_factory=list,
        max_length=5,
        description="Research gaps identified across the papers.",
    )

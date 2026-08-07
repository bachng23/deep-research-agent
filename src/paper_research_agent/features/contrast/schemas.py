from pydantic import BaseModel, Field

from paper_research_agent.core.state import ResearchGap


class GapAnalysis(BaseModel):
    gaps: list[ResearchGap] = Field(
        default_factory=list,
        # Discovery is capped at 5 by the prompt; the refine pass is allowed to
        # carry those 5 forward and add 1-2. A cap of 5 here made that a schema
        # violation -> 3 wasted retries -> gaps silently left unrefined.
        max_length=8,
        description="Research gaps identified across the papers.",
    )

from pydantic import BaseModel, Field

from paper_research_agent.core.state import Conflict


class ConflictAnalysis(BaseModel):
    conflicts: list[Conflict] = Field(
        default_factory=list,
        max_length=5,
        description="Genuine disagreements between the papers.",
    )

from typing import Literal

from pydantic import BaseModel, Field

from paper_research_agent.core.models import Paper

Confidence = Literal["low", "medium", "high"]


class ResearchGap(BaseModel):
    description: str
    supporting_papers: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(
        default_factory=list,
        description="Verbatim sentences copied exactly from the abstracts that justify this gap. No paraphrasing.",
    )
    confidence: Confidence = "medium"


class ResearchState(BaseModel):
    topic: str
    user_idea: str | None = None

    search_queries: list[str] = Field(default_factory=list)
    papers: list[Paper] = Field(default_factory=list)

    gaps: list[ResearchGap] = Field(default_factory=list)

    novelty_score: int | None = Field(default=None, ge=0, le=100)
    novelty_reasoning: str | None = None
    overlapping_papers: list[str] = Field(default_factory=list)

    report_markdown: str | None = None

    errors: list[str] = Field(default_factory=list)
    tool_call_count: int = 0

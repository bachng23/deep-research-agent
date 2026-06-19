import time
from typing import Literal

from pydantic import BaseModel, Field

from paper_research_agent.core.models import Paper

Confidence = Literal["low", "medium", "high"]


class ResearchGap(BaseModel):
    description: str
    supporting_papers: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(
        default_factory=list,
        description="Verbatim sentences copied exactly from the provided text that justify this gap. No paraphrasing.",
    )
    confidence: Confidence = "medium"


class RoundLog(BaseModel):
    "One interation of the research loop, for inspection."

    iteration: int
    queries: list[str] = Field(default_factory=list)
    new_papers: int = 0
    total_papers: int = 0
    open_gaps: int = 0
    closed_gaps: int = 0


class Conflict(BaseModel):
    topic: str = Field(description="The specific point the papers disagree on.")

    position_a: str
    position_a_papers: list[str] = Field(default_factory=list)
    position_a_quote: str = ""

    position_b: str
    position_b_papers: list[str] = Field(default_factory=list)
    position_b_quote: str = ""


class ResearchState(BaseModel):
    topic: str
    user_idea: str | None = None

    search_queries: list[str] = Field(default_factory=list)
    papers: list[Paper] = Field(default_factory=list)

    gaps: list[ResearchGap] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    recalled_gaps: list[str] = Field(default_factory=list)

    novelty_score: int | None = Field(default=None, ge=0, le=100)
    novelty_reasoning: str | None = None
    overlapping_papers: list[str] = Field(default_factory=list)

    report_markdown: str | None = None

    # research loop memory
    iteration: int = 0
    max_iterations: int = 3
    timeout_seconds: float | None = None
    started_at: float = Field(default_factory=time.monotonic)
    seen_paper_ids: list[str] = Field(default_factory=list)
    recent_paper_ids: list[str] = Field(default_factory=list)
    open_gaps: list[str] = Field(default_factory=list)
    round_logs: list[RoundLog] = Field(default_factory=list)

    use_memory: bool = False

    use_llm_judge: bool = True
    coverage_sufficient: bool = False
    coverage_reasoning: str | None = None

    use_tool_agent: bool = True
    read_full_text: bool = False

    errors: list[str] = Field(default_factory=list)
    tool_call_count: int = 0

from pydantic import BaseModel, Field


class NoveltyAssessment(BaseModel):
    score: int = Field(
        ge=0,
        le=100,
        description="Novelty of the user's idea: 100 = entirely new, 0 = already fully covered by existing papers.",
    )

    reasoning: str = Field(
        description="Concise justification grounded in the papers and gaps."
    )

    overlapping_papers: list[str] = Field(
        default_factory=list,
        description="Exact titles of papers that already cover part of the idea.",
    )

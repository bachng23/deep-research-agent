from pydantic import BaseModel, Field


class CoverageJudgment(BaseModel):
    sufficient: bool = Field(
        description="True if another search round would NOT materially improve the coverage needed to assess the user's idea."
    )
    reasoning: str = Field(description="Brief justification for the decision.")

from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    queries: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Search queries for academic paper retrieval.",
    )

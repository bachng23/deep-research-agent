from typing import Literal

from pydantic import BaseModel, Field

PaperSource = Literal["arxiv", "openalex"]


class Paper(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    url: str | None = None
    source: PaperSource
    citation_count: int | None = None
    pdf_url: str | None = None
    full_text_excerpt: str | None = None

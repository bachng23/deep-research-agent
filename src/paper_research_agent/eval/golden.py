from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

_DATA_FILE = Path(__file__).parent / "data" / "golden_set.json"

GapSource = Literal["section", "abstract", "debate"]


class Provenance(BaseModel):
    """Where a case's ground truth came from, so a reader can check it.

    Kept alongside the labels rather than in a separate doc: ground truth whose
    origin is not auditable is not ground truth.
    """

    survey_title: str
    arxiv_id: str
    gap_source: str
    gap_note: str = ""
    paper_source: str = ""

    @property
    def granularity(self) -> GapSource:
        "How strong the gap labels are: an enumerated section beats an abstract."
        if self.gap_source.startswith("section"):
            return "section"
        if self.gap_source.startswith("debate"):
            return "debate"
        return "abstract"


class GoldenCase(BaseModel):
    topic: str
    idea: str | None = None

    expected_gap_keywords: list[str] = Field(default_factory=list)
    expected_papers: list[str] = Field(default_factory=list)
    min_gaps: int = Field(default=1, ge=0)

    difficulty: Literal["narrow", "broad"] = "broad"
    # True only where the literature genuinely disagrees, so "no conflicts
    # found" is a real miss rather than a correct answer on a settled topic.
    expects_conflict: bool = False

    provenance: Provenance | None = None


@lru_cache
def load_golden(path: Path | None = None) -> tuple[GoldenCase, ...]:
    raw = json.loads((path or _DATA_FILE).read_text(encoding="utf-8"))
    return tuple(GoldenCase.model_validate(c) for c in raw["cases"])


GOLDEN: list[GoldenCase] = list(load_golden())

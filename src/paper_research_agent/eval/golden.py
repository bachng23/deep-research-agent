from __future__ import annotations

from pydantic import BaseModel, Field


class GoldenCase(BaseModel):
    topic: str
    idea: str | None = None

    expected_gap_keywords: list[str] = Field(default_factory=list)
    expected_papers: list[str] = Field(default_factory=list)
    min_gaps: int = Field(default=1, ge=0)


GOLDEN: list[GoldenCase] = [
    GoldenCase(
        topic="hierarchical chunking for retrieval augmented generation over long scientific documents",
        idea="use document structure (sections, tables) to build hierarchical chunks",
        expected_gap_keywords=["table", "structure", "evaluation", "long document"],
        expected_papers=["RAPTOR", "dense passage"],
        min_gaps=2,
    ),
    GoldenCase(
        topic="retrieval-augmented generation versus long-context language models",
        idea="use retrieval to pre-filter passages before a long-context model",
        expected_gap_keywords=["long context", "retrieval", "evaluation", "cost"],
        expected_papers=["Lost in the Middle", "Retrieval meets Long Context"],
        min_gaps=2,
    ),
    GoldenCase(
        topic="parameter-efficient fine-tuning of large language models",
        idea="combine low-rank adapters with quantization for cheaper fine-tuning",
        expected_gap_keywords=[
            "memory",
            "catastrophic forgetting",
            "evaluation",
            "task",
        ],
        expected_papers=["LoRA", "QLoRA", "Prefix-Tuning"],
        min_gaps=2,
    ),
    GoldenCase(
        topic="tool use and function calling in large language model agents",
        idea="let the agent learn when NOT to call a tool to reduce error cascades",
        expected_gap_keywords=["reliability", "error", "planning", "benchmark"],
        expected_papers=["Toolformer", "ReAct"],
        min_gaps=2,
    ),
    GoldenCase(
        topic="hallucination detection and mitigation in large language models",
        idea="use retrieval grounding signals to flag likely hallucinations at decode time",
        expected_gap_keywords=["factuality", "detection", "evaluation", "retrieval"],
        expected_papers=["SelfCheckGPT", "Survey of Hallucination"],
        min_gaps=2,
    ),
    GoldenCase(
        topic="preference optimization methods for aligning language models",
        idea="reduce the reward-model dependence of RLHF with direct preference signals",
        expected_gap_keywords=["reward", "stability", "data", "evaluation"],
        expected_papers=[
            "Direct Preference Optimization",
            "Proximal Policy Optimization",
        ],
        min_gaps=2,
    ),
]

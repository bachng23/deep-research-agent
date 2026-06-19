from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from paper_research_agent.llm import chat_model_for_tier, invoke_with_retry


class Intent(BaseModel):
    action: Literal["research", "qa"]


_ROUTER_SYSTEM = """Classify the user's message into one action.

"research" — names a TOPIC / subject area to investigate from scratch
  (usually a noun phrase, NOT a question). Starts a multi-minute run.
  Examples:
    "retrieval augmented generation over long documents" -> research
    "transformer attention efficiency" -> research
    "graph neural networks for drug discovery" -> research

"qa" — asks a QUESTION to be answered from papers already read
  (interrogative, or asks for a specific fact / summary / comparison).
  Examples:
    "what causes hallucination in RAG?" -> qa
    "which paper said long-context beats retrieval?" -> qa
    "summarize the open gaps" -> qa

A bare topic/subject with no question -> "research"."""


def route(text: str) -> str:
    "Classify a message as 'research' or 'qa' (fast LLM) safe fallback 'qa'."
    try:
        model = chat_model_for_tier("fast").with_structured_output(Intent)
        result = invoke_with_retry(model, [("system", _ROUTER_SYSTEM), ("user", text)])
        return result.action
    except Exception:
        return "qa"

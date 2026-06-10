from __future__ import annotations

from paper_research_agent.core.state import ResearchState
from paper_research_agent.features.planning.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
)
from paper_research_agent.features.planning.schemas import QueryPlan
from paper_research_agent.llm import chat_model_for_tier


def plan_queries(state: ResearchState) -> ResearchState:
    if state.search_queries:
        return state

    try:
        state.search_queries = _plan_queries_with_llm(state)
    except Exception as e:
        state.errors.append(f"planning failed: {e}")
        state.search_queries = [state.topic]

    return state


def _plan_queries_with_llm(state: ResearchState) -> list[str]:
    model = chat_model_for_tier("fast")
    structured_model = model.with_structured_output(QueryPlan)

    user_idea = state.user_idea or "Not provided"

    prompt = PLANNER_USER_PROMPT.format(topic=state.topic, user_idea=user_idea)

    result = structured_model.invoke(
        [("system", PLANNER_SYSTEM_PROMPT), ("user", prompt)]
    )

    return _clean_queries(result.queries, fallback=state.topic)


def _clean_queries(queries: list[str], *, fallback: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()

    for query in queries:
        normalized = " ".join(query.strip().split())
        key = normalized.lower()

        if not normalized:
            continue

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(normalized)

    return cleaned or [fallback]

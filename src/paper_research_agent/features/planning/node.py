from __future__ import annotations

from paper_research_agent.core.state import ResearchState
from paper_research_agent.features.planning.prompts import (
    FOLLOWUP_USER_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
)
from paper_research_agent.features.planning.schemas import QueryPlan
from paper_research_agent.llm import chat_model_for_tier, invoke_with_retry


def plan_queries(state: ResearchState) -> ResearchState:
    # no queries yet -> plan from topic
    if state.search_queries and not state.open_gaps:
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

    if state.open_gaps:
        prompt = FOLLOWUP_USER_PROMPT.format(
            topic=state.topic,
            open_gaps="\n".join(f"- {gap}" for gap in state.open_gaps),
        )
    else:
        prompt = PLANNER_USER_PROMPT.format(
            topic=state.topic,
            user_idea=state.user_idea or "Not provided",
        )

    result = invoke_with_retry(
        structured_model, [("system", PLANNER_SYSTEM_PROMPT), ("user", prompt)]
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

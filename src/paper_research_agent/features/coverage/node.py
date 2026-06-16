from __future__ import annotations

import time

from paper_research_agent.core.state import ResearchGap, ResearchState, RoundLog
from paper_research_agent.features.coverage.prompts import (
    COVERAGE_SYSTEM_PROMPT,
    COVERAGE_USER_PROMPT,
)
from paper_research_agent.features.coverage.schemas import CoverageJudgment
from paper_research_agent.llm import chat_model_for_tier, invoke_with_retry


def assess_coverage(state: ResearchState) -> ResearchState:
    state.iteration += 1

    open_gaps = [g.description for g in state.gaps if g.confidence != "high"]
    closed_gaps = len(state.gaps) - len(open_gaps)
    state.open_gaps = open_gaps

    prev_total = state.round_logs[-1].total_papers if state.round_logs else 0
    state.round_logs.append(
        RoundLog(
            iteration=state.iteration,
            queries=list(state.search_queries),
            new_papers=len(state.papers) - prev_total,
            total_papers=len(state.papers),
            open_gaps=len(open_gaps),
            closed_gaps=closed_gaps,
        )
    )

    return state


def judge_coverage(state: ResearchState) -> ResearchState:
    if not state.use_llm_judge or not state.gaps:
        return state

    try:
        judgment = _judge_with_llm(state)
        state.coverage_sufficient = judgment.sufficient
        state.coverage_reasoning = judgment.reasoning
    except Exception as e:
        state.errors.append(f"coverage judge failed: {e}")

    return state


def _judge_with_llm(state: ResearchState) -> CoverageJudgment:
    model = chat_model_for_tier("fast").with_structured_output(CoverageJudgment)

    prompt = COVERAGE_USER_PROMPT.format(
        topic=state.topic,
        user_idea=state.user_idea or "Not provided",
        gaps=_format_gaps_for_judge(state.gaps),
    )

    return invoke_with_retry(
        model, [("system", COVERAGE_SYSTEM_PROMPT), ("user", prompt)]
    )


def _format_gaps_for_judge(gaps: list[ResearchGap]) -> str:
    return "\n".join(
        f"- [{g.confidence}] {g.description} "
        f"(supported by {len(g.supporting_papers)} papers)"
        for g in gaps
    )


def should_continue(state: ResearchState) -> bool:
    """Stop if: no open gaps, ceiling reached, timed out, OR the LLM judge
    deems coverage sufficient. The judge can only stop EARLIER, never override
    the hard caps."""
    if not state.open_gaps:
        return False

    if state.iteration >= state.max_iterations:
        return False

    if state.timeout_seconds is not None:
        elapsed = time.monotonic() - state.started_at
        if elapsed >= state.timeout_seconds:
            return False

    if state.coverage_sufficient:
        return False

    return True

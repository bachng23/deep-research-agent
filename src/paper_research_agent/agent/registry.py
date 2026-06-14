from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from paper_research_agent.core.state import ResearchState
from paper_research_agent.features.contrast import find_gaps
from paper_research_agent.features.fetching import fetch_papers
from paper_research_agent.features.planning import plan_queries

NodeFn = Callable[[ResearchState], ResearchState]


@dataclass(frozen=True)
class NodeSpec:
    name: str
    run: NodeFn


def default_nodes() -> list[NodeSpec]:
    return [
        NodeSpec("plan_queries", plan_queries),
        NodeSpec("fetch_papers", fetch_papers),
        NodeSpec("find_gaps", find_gaps),
    ]

from __future__ import annotations

from langgraph.graph import END, StateGraph

from paper_research_agent.agent.nodes import fetch_papers
from paper_research_agent.state import ResearchState


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("fetch_papers", fetch_papers)

    graph.set_entry_point("fetch_papers")
    graph.add_edge("fetch_papers", END)


def run_research(topic: str, user_idea: str | None = None) -> ResearchState:
    app = build_graph()

    initial_state = ResearchState(topic=topic, user_idea=user_idea)

    return app.invoke(initial_state)

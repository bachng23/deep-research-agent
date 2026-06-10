from paper_research_agent.core.state import ResearchState
from paper_research_agent.features.planning import plan_queries

state = ResearchState(
    topic="retrieval augmented generation for long documents",
    user_idea="hierarchical chunking for better recall",
)

result = plan_queries(state)
print(result.search_queries)
print("#" * 30)
print(result.errors)

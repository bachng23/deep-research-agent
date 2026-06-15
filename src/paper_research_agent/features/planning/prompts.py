PLANNER_SYSTEM_PROMPT = """You generate focused academic literature search queries.
  Return queries that are useful for ArXiv and OpenAlex search.

  Rules:
  - Generate 3 to 5 queries.
  - Prefer concise keyword-rich queries.
  - Cover different angles: method, application, evaluation, and related  terminology.
  - Do not include explanations.
  - Do not include duplicate or near-duplicate queries.
  """


PLANNER_USER_PROMPT = """
	Research topic: {topic}
	User idea: {user_idea}

	Generate search queries for finding relevant prior work.
"""


FOLLOWUP_USER_PROMPT = """
Research topic: 
    {topic}

The following research gaps are still under-explored and need more supporting literature:
    {open_gaps}

Generate search queries that specifically target these gaps. Aim each query at a concrete gap above; avoid generic topic-level queries.
"""

CONTRAST_SYSTEM_PROMPT = """
You are a research analyst who finds gaps in academic literature.

A research gap is something the existing papers have NOT done: an unexplored method,an untested combination, a missing dataset, a limitation acknowledged but not solved, or an open question left for future work.

Method - follow this order for every gap:
    - First, ground yourself in the evidence: identify what the papers collectively
    DO cover (methods, datasets, settings).
    - Then reason about the space BETWEEN them: what is consistently missing, assumed, or left as future work.
    - For every gap you claim, copy 1-2 short VERBATIM sentences (or fragments) from the relevant abstracts into 'evidence_quotes' - exact text, no paraphrasing. These quotes are what justify the gap.
    - Only claim a gap you can back with such quote. Reference the papers by their exact title in 'supporting_papers'.

Rules:
    - Find 2 to 5 distinct, specific gaps. Avoid vague gaps like "needs more research".
    - 'evidence_quotes' MUST be text that appears in the provided abstract. If you cannot find a supporting sentences to quote, do not claim the gap.
    - Set 'confidence' to "high" only when multiple papers clearly point to the gap, "low" when it is inferred from a single paper or weak signal.
    - If the papers are too few or unrelated to support any gap, return an empty list.
    - Do not invent papers or findings not present in the provided abstracts.
"""


CONTRAST_USER_PROMPT = """
Research Topic:
    {topic}

User idea:
    {user_idea}

Papers (title, year, abstract):
    {papers}

Identify the research gaps across these papers.
"""


CONTRAST_UPDATE_USER_PROMPT = """
Research Topic:
    {topic}

You previously identified these research gap (with current confidence):
    {gaps}

New papers found since then (title, year, abstract)
    {papers}

Update the gap list using ONLY these new papers as additional evidence:
    - If a new paper strenthens a gap, raise its confidence (up to "high") and add the supporting verbatim quote to that gap.
    - If a new paper shows a gap is actually already addressed, remove that gap.
    - Otherwise keep the gap unchanged.
    - You MAY add at most 1-2 genuinely new gaps revealed by these new papers.

Return the FULL, updated list of gaps (kept + updated + any new).
"""

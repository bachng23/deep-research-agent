from __future__ import annotations

from paper_research_agent.config import get_settings
from paper_research_agent.core import Paper
from paper_research_agent.features.qa.prompts import QA_SYSTEM_PROMPT
from paper_research_agent.llm import chat_model_for_tier
from paper_research_agent.memory import PaperMemory


def answer_question(question: str, *, top_k: int = 8) -> str:
    "Q&A over the agent's long-term paper memory (cross-session, grounded)."
    memory = PaperMemory(get_settings().memory_dir)
    try:
        hits = memory.relevant(question, top_k=top_k)
    finally:
        memory.close()

    if not hits:
        return "No paper in memory yet - research a topic first (with memory on)."

    context = "\n\n".join(
        f"[{h.get('paper_id', '?')} . {h.get('section') or 'Body'}]\n{h['text']}"
        for h in hits
    )
    model = chat_model_for_tier("balanced")
    resp = model.invoke(
        [
            ("system", QA_SYSTEM_PROMPT),
            ("user", f"Passages:\n{context}\n\nQuestion:{question}"),
        ]
    )
    return str(resp.content).strip()

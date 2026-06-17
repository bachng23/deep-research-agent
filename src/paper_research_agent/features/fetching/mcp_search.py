from __future__ import annotations

import asyncio
import json

from fastmcp import Client
from langchain_mcp_adapters.tools import load_mcp_tools

from paper_research_agent.core.models import Paper
from paper_research_agent.llm import chat_model_for_tier
from paper_research_agent.mcp.server import mcp

FETCH_SYSTEM_PROMPT = """
   You find academic papers by calling search tools. Pick the right tool per
   query: arxiv_search for recent CS/ML/AI/DL/physics preprints, openalex_search for established, highly-cited, or cross-disciplinary work. Issue one or more tool calls covering the queries; use both tools when it improves coverage.
   """


def search_via_tool_agent(queries: list[str]) -> tuple[list[Paper], int]:
    """
    Let the LLM choose and call the MCP search tool.
    Sync wrapper around the async MCP client (safe: LangGraph nodes run outside an event loop).
    """
    return asyncio.run(_search_via_tool_agent(queries))


async def _search_via_tool_agent(queries: list[str]) -> tuple[list[Paper], int]:
    async with Client(mcp) as client:
        tools = await load_mcp_tools(client.session)
        by_name = {t.name: t for t in tools}

        model = chat_model_for_tier("fast").bind_tools(tools)
        prompt = "Find papers for these queries:\n" + "\n".join(
            f"- {q}" for q in queries
        )

        message = await model.ainvoke(
            [
                ("system", FETCH_SYSTEM_PROMPT),
                ("user", prompt),
            ]
        )

        papers: list[Paper] = []
        n_calls = 0
        for call in message.tool_calls:
            tool = by_name.get(call["name"])
            if tool is None:
                continue
            blocks = await tool.ainvoke(call["args"])
            papers.extend(_parse_papers(blocks))
            n_calls += 1

    return papers, n_calls


def _parse_papers(blocks) -> list[Paper]:
    """MCP tools return content blocks; the text block holds a JSON array of paper dicts matching the Paper schema."""
    papers: list[Paper] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            for item in json.loads(block["text"]):
                papers.append(Paper(**item))
    return papers

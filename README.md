# paper-research-agent

An AI agent that researches academic literature (ArXiv + OpenAlex), reads paper full text, identifies research gaps, surfaces conflicting evidence, and scores the novelty of your idea — built with LangGraph, model-agnostic via any OpenAI-compatible endpoint (OpenRouter by default).

---

## What it does

Given a topic (and optionally your own idea), the agent runs an iterative research loop:

1. **Plans** search queries from the topic — and on later rounds, from the gaps still open
2. **Fetches** papers from ArXiv + OpenAlex, deduplicates and ranks them
3. **Finds gaps** across paper abstracts, each gap tied to supporting papers and verbatim quotes
4. **Judges its own coverage** and decides whether to search again or stop
5. **Reads full text** of the top papers (ArXiv HTML → PDF fallback), chunked and indexed in Qdrant
6. **Refines gaps** against what the full text actually says — pruning false gaps, adding grounded ones
7. **Detects conflicts** — where papers disagree, with the quote from each side
8. **Scores novelty** of your idea against the literature (0–100) and names overlapping papers
9. **Writes** a cited Markdown report

It also remembers across sessions: papers read stay in a persistent index, past gaps get recalled on related topics, and a semantically equivalent topic researched recently is served from cache instead of re-run.

---

## Pipeline

```
                    recall_prior        prior gaps on related topics   [no LLM]
                          │
                  ┌───────▼────────┐
        ┌────────►│  plan_queries   │  topic/open gaps → queries       [fast]
        │         └───────┬────────┘
        │         ┌───────▼────────┐
        │         │  fetch_papers   │  ArXiv + OpenAlex, dedup, rank   [fast, tool-calling]
        │         └───────┬────────┘
        │         ┌───────▼────────┐
        │         │   find_gaps     │  gaps from abstracts             [reasoning]
        │         └───────┬────────┘
        │         ┌───────▼────────┐
        │         │ assess_coverage │  round bookkeeping               [no LLM]
        │         └───────┬────────┘
        │         ┌───────▼────────┐
        │         │ judge_coverage  │  "have we seen enough?"          [fast]
        │         └───────┬────────┘
        │                 │
        └── continue ─────┤ should_continue()
                          │
                       finish
                          │
                  ┌───────▼────────┐
                  │  read_papers    │  full text → chunk → Qdrant      [embeddings]
                  └───────┬────────┘
                  ┌───────▼────────┐
                  │  refine_gaps    │  re-check gaps vs full text      [reasoning]
                  └───────┬────────┘
                  ┌───────▼────────┐
                  │   rank_gaps     │  importance ordering             [no LLM]
                  └───────┬────────┘
                  ┌───────▼────────┐
                  │ find_conflicts  │  disagreements + quotes          [reasoning]
                  └───────┬────────┘
                  ┌───────▼────────┐
                  │  score_novelty  │  idea vs literature (0–100)      [reasoning]
                  └───────┬────────┘
                  ┌───────▼────────┐
                  │  write_report   │  cited Markdown                  [balanced]
                  └───────┬────────┘
                    remember_result   persist topic → gaps             [fast, embeddings]
```

The graph is assembled in [`agent/graph.py`](src/paper_research_agent/agent/graph.py); each node is a vertical slice under `features/<name>/` communicating only through `ResearchState`.

### The research loop

`should_continue()` ([coverage/node.py](src/paper_research_agent/features/coverage/node.py)) stops when **any** of: no gaps left open, `max_iterations` reached, timeout exceeded, or the LLM coverage judge says coverage is sufficient. The judge can only stop the loop **earlier** — it can never override the hard caps. This keeps a bounded worst case while letting an easy topic finish in one round.

### Grounding

Gaps carry `evidence_quotes` — verbatim sentences the model must copy, not paraphrase. After full-text reading, `refine_gaps` re-examines every gap against Results/Discussion sections, so a gap that only looked open in the abstracts gets pruned. Chunking tags sections and flags Results/Discussion (`is_findings`), where claims and contradictions actually live.

---

## Model routing

Nodes never name a model — they request a capability **tier**, resolved in `config.py` and instantiated in `llm.py`. All calls run at `temperature=0.0`.

| Tier | Default model | Used by |
|---|---|---|
| fast | `deepseek/deepseek-v4-flash` | Planner, coverage judge, fetch tool-agent, intent router, topic normalization |
| balanced | `deepseek/deepseek-v4-pro` | Writer, Q&A |
| reasoning | `deepseek/deepseek-v4-pro` | Gap discovery, gap refinement, conflicts, novelty |

Served through any OpenAI-compatible endpoint (`LLM_BASE_URL`). Set `MODEL_TIER_OVERRIDE` to force every node onto one tier for A/B testing.

---

## Interfaces

### TUI

```bash
uv run paper-research
```

A Textual app with three modes — `ctrl+t` cycles **auto / research / Q&A**. In auto mode an LLM intent router ([`agent/router.py`](src/paper_research_agent/agent/router.py)) classifies each message: a bare topic starts a research run, a question is answered from papers already read. The router falls back to Q&A on any error. Runs stream node-by-node with live counts. Memory is on in the TUI.

### REST API

```bash
uv run uvicorn paper_research_agent.api.main:app --reload --port 8000
```

`POST /research` — body `{"topic": "...", "user_idea": "..."}` (topic min length 3). Returns:

```json
{
  "topic": "retrieval-augmented generation for long documents",
  "papers_found": 16,
  "papers": [{ "title": "…", "authors": ["…"], "year": 2024, "abstract": "…",
               "url": "https://arxiv.org/abs/…", "source": "arxiv", "citation_count": 41 }],
  "gaps": [{ "description": "…", "supporting_papers": ["…"],
             "evidence_quotes": ["…"], "confidence": "medium" }],
  "novelty_score": 62,
  "novelty_reasoning": "…",
  "overlapping_papers": ["…"],
  "report_markdown": "## Overview\n…"
}
```

`GET /health` → `{"status": "ok"}`

### MCP server

```bash
uv run python -m paper_research_agent.mcp.server
```

Exposes `arxiv_search` and `openalex_search` as FastMCP tools. The fetch node consumes these itself via a tool-calling agent (`use_tool_agent`), letting the LLM choose the source per query; it falls back to deterministic multi-provider search if the tool agent returns nothing.

### Python

```python
from paper_research_agent import run_research

state = run_research("retrieval augmented generation over long documents",
                     user_idea="hierarchical chunking using document structure",
                     read_full_text=True, max_iterations=2)
print(state.report_markdown)
```

---

## Evaluation

The eval harness lives in [`src/paper_research_agent/eval/`](src/paper_research_agent/eval/): a golden set of research topics with expected papers and gap keywords, metrics over a finished run (`grounded_in_fulltext`, `gap_keyword_recall`, `paper_recall`, `citation_coverage`, plus operational counters), heuristic failure analysis, and a runner supporting repeated runs for variance.

```bash
uv run python -m paper_research_agent.eval.runner --repeats 3 --max-iterations 2
```

> **Status: not yet a reportable result.** An audit of the harness found `citation_coverage` measures nothing (a regex bug makes it constant), `grounded_in_fulltext` conflates "no data" with "0% grounded", the golden set is 6 hard-coded topics with no recorded provenance, and runs persist no model/commit/date metadata. No evaluation numbers are published here yet because none of them would currently survive scrutiny. See `eval/README.md` once the harness is fixed and a real run is committed.

---

## Project structure

Dependency direction: `core` ← `providers` ← `features` ← `agent` ← (`api`, `mcp`, `tui`).
Lower layers never import higher ones; features never import each other — they communicate only through `ResearchState`.

```
src/paper_research_agent/
├── config.py               # Pydantic Settings + tier → model mapping
├── llm.py                  # chat model / embeddings factory, invoke_with_retry
├── tui.py                  # Textual app (`paper-research`)
│
├── core/                   # domain layer — depends on nothing else
│   ├── models.py           # Paper, PaperSource
│   ├── state.py            # ResearchState, ResearchGap, Conflict, RoundLog
│   ├── errors.py           # ProviderError, RateLimitError, …
│   └── logging.py
│
├── providers/              # external paper sources
│   ├── base.py             # PaperProvider protocol — add a source = 1 file
│   ├── arxiv.py
│   └── openalex.py
│
├── features/               # vertical slices — one package per graph node
│   ├── planning/           # plan_queries (LLM query expansion + follow-ups)
│   ├── fetching/           # fetch_papers: tool-agent + service, dedup, ranking
│   ├── contrast/           # find_gaps, refine_gaps, rank_gaps
│   ├── coverage/           # assess_coverage, judge_coverage, should_continue
│   ├── reading/            # fetch full text, chunk, index, build excerpts
│   ├── conflicts/          # find_conflicts
│   ├── novelty/            # score_novelty
│   ├── writing/            # write_report
│   └── qa/                 # answer_question over paper memory
│
├── memory/                 # cross-session memory
│   ├── store.py            # PaperMemory — persistent full-text index
│   ├── results.py          # ResultMemory — topic → gaps, semantic cache
│   └── nodes.py            # recall_prior, remember_result
│
├── agent/                  # graph assembly + intent routing
│   ├── graph.py            # StateGraph, run_research(), stream_research()
│   └── router.py           # research vs Q&A intent classifier
│
├── api/                    # FastAPI layer
├── mcp/server.py           # FastMCP tools: arxiv_search, openalex_search
└── eval/                   # golden set, metrics, runner, failure analysis
```

---

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent framework | LangGraph | Stateful graph with a conditional research loop |
| LLM access | `langchain-openai` → OpenRouter | One OpenAI-compatible client, tier-routed models |
| Data — papers | ArXiv API, OpenAlex API | Metadata, abstracts, citation counts |
| Full text | trafilatura (ArXiv HTML), PyMuPDF (PDF) | Read beyond the abstract |
| Vector index | Qdrant (embedded) | Chunk-level retrieval over read papers |
| Tools | FastMCP | Paper search exposed as MCP tools |
| Schemas & config | Pydantic / pydantic-settings | State, domain models, typed settings |
| API / TUI | FastAPI + Uvicorn / Textual | REST endpoint, terminal app |
| Observability | LangSmith (optional) | Tracing |
| Tooling | uv, pytest, ruff | Env management, tests, lint |

---

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- An API key for an OpenAI-compatible endpoint (default: [OpenRouter](https://openrouter.ai/keys))
- (Optional) An [OpenAlex API key](https://openalex.org/) — raises rate limits
- (Optional) A [LangSmith API key](https://smith.langchain.com/) for tracing

### Install

```bash
git clone https://github.com/bachng23/deep-re-search-agent
cd deep-re-search-agent
uv sync
```

Create a `.env`:

```bash
# Required — key for the OpenAI-compatible LLM endpoint
API_KEY=

# Optional — endpoint base URL (default: https://openrouter.ai/api/v1)
LLM_BASE_URL=

# Optional — force every node onto one tier: "fast" | "balanced" | "reasoning"
MODEL_TIER_OVERRIDE=

# Optional — cross-session memory (the TUI enables it regardless)
USE_MEMORY=false
MEMORY_DIR=.paper_research_memory
RESULT_CACHE_TTL_DAYS=7

# Optional — reading / chunking
READ_MAX_PAPERS=5
FULLTEXT_CHUNK_CHARS=1500
FULLTEXT_TOP_K=5

# Optional — provider limits
ARXIV_MAX_RESULTS=8
OPENALEX_MAX_RESULTS=8
REQUEST_TIMEOUT_SECONDS=20
OPENALEX_API_KEY=

# Optional — LangSmith tracing
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=paper-research-agent
```

### Test

```bash
uv run pytest                   # unit tests, no network (91 tests)
uv run pytest -m integration    # hits real ArXiv/OpenAlex APIs
uv run ruff check src tests     # lint
```

---

## Design principles

| Principle | Where |
|---|---|
| **Schema-first state** — Pydantic models define the contract before any prompt exists | `core/state.py` |
| **Tier routing, not model names** — nodes request a capability; mapping lives in one place | `config.py`, `llm.py` |
| **Provider protocol** — adding a paper source is one file implementing `PaperProvider` | `providers/base.py` |
| **Extraction vs calculation** — dedup, ranking, chunking are pure Python, not LLM calls | `features/fetching/`, `features/reading/` |
| **Quote-first prompting** — gaps must carry verbatim evidence, checked against full text | `features/contrast/` |
| **Bounded agency** — the LLM judge may stop the loop early but never past the hard caps | `features/coverage/` |
| **Graceful degradation** — full-text read failure falls back to abstracts; tool-agent failure falls back to deterministic search; router failure falls back to Q&A | `features/reading/`, `features/fetching/`, `agent/router.py` |
| **Helpful errors as prompts** — rate limits raise `"Wait or set OPENALEX_API_KEY"`, not a bare 429 | `providers/openalex.py` |
| **Tests without network by default** — the graph runs end-to-end against stub providers | `tests/` |

---

## Known limitations

- **Full-text reading is best-effort**: ArXiv HTML when available, PDF otherwise; non-ArXiv papers behind paywalls degrade to abstract-only. Only the top `READ_MAX_PAPERS` (default 5) are read.
- **Novelty is an LLM self-report** — a 0–100 score with reasoning, not a validated measurement. Treat it as a prompt for your own judgement.
- **English / CS bias**: ArXiv and OpenAlex coverage is strongest for English-language CS papers.
- **No published eval numbers** — see the Evaluation section for why.
- **Rate limits**: OpenAlex throttles anonymous usage. Set `OPENALEX_API_KEY` for heavy use.
- **Semantic result cache can serve a stale run** — a topic within `RESULT_CACHE_TTL_DAYS` and above the similarity threshold short-circuits the graph entirely.

---

## Roadmap

- [ ] Fix and re-baseline the eval harness (see Evaluation), publish `eval/README.md`
- [ ] Token/cost accounting per run
- [ ] PDF upload — read your own draft, agent finds gaps in your contribution
- [ ] Citation graph traversal via OpenAlex references
- [ ] Web demo frontend

---

## Built for

AIE1 Final Project — Module 1: Building a complete AI Agent

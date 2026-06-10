# paper-research-agent

An AI agent that scans academic literature (ArXiv + OpenAlex), identifies research gaps, and evaluates the novelty of your ideas — built with LangGraph and FastAPI, model-agnostic via an OpenAI-compatible endpoint (OpenRouter by default).

---

## Status

The pipeline skeleton runs end-to-end (CLI, REST API, tests), with the literature-fetching stage fully implemented. The LLM-powered stages are next.

| Stage | Node | Status |
|---|---|---|
| Planner | `plan_queries` | ⚙️ Deterministic fallback (topic as single query) — LLM planner planned |
| Fetcher | `fetch_papers` | ✅ ArXiv + OpenAlex search, dedup, ranking |
| Contrast | `find_gaps` | 🔜 Planned |
| Novelty | `score_novelty` | 🔜 Planned |
| Writer | `write_report` | 🔜 Planned |

---

## What it will do

Given a research topic or idea, the agent:

1. **Searches** ArXiv and OpenAlex for relevant papers ✅
2. **Contrasts** methodologies, datasets, and findings across papers to surface what's missing
3. **Scores** your idea's novelty against existing literature (0–100)
4. **Reports** identified gaps with citations, and explains why your idea is or isn't novel

---

## Architecture

```
User input (topic or idea)
        │
   ┌────▼─────┐
   │  Planner  │  Expands topic into search queries          [fast tier]
   └────┬─────┘  (currently deterministic: topic as-is)
        │
   ┌────▼─────┐
   │  Fetcher  │  Searches ArXiv + OpenAlex, dedups, ranks   [no LLM]
   └────┬─────┘
        │
   ┌────▼─────┐
   │  Contrast │  Finds gaps between papers (planned)        [reasoning tier]
   └────┬─────┘
        │
   ┌────▼──────────┐
   │ Novelty checker│  Scores your idea vs gaps (planned)    [reasoning tier]
   └────┬──────────┘
        │
   ┌────▼─────┐
   │  Writer   │  Synthesizes cited report (planned)         [balanced tier]
   └────┬─────┘
        │
   Research report (JSON + Markdown)
```

Nodes are declared as an ordered list in `agent/registry.py`; `agent/graph.py` wires whatever the registry declares into a LangGraph `StateGraph`. Adding a pipeline stage = implement `features/<name>/node.py` + add one `NodeSpec`.

The **Contrast node** is what will make this agent different from a literature summarizer: it reasons about the *space between papers* to identify what hasn't been done.

---

## Model routing

Nodes never name a model — they request a capability **tier** (`fast` / `balanced` / `reasoning`), resolved in `config.py` and instantiated in `llm.py`. Defaults (override via env):

| Tier | Default model | Used by |
|---|---|---|
| fast | `deepseek/deepseek-v4-flash` | Planner |
| balanced | `deepseek/deepseek-v4-pro` | Writer |
| reasoning | `deepseek/deepseek-v4-pro` | Contrast, Novelty |

Models are served through any OpenAI-compatible endpoint (`LLM_BASE_URL`, default OpenRouter). Set `MODEL_TIER_OVERRIDE` to force every node onto one tier for A/B testing.

---

## Project structure

Dependency direction: `core` ← `providers` ← `features` ← `agent` ← (`api`, `cli`).
Lower layers never import higher ones; features never import each other —
they communicate only through `ResearchState`.

```
paper-research-agent/
├── pyproject.toml                        # deps, `paper-research` CLI entry point, pytest config
├── .env.example                          # environment variables
├── README.md
│
├── src/paper_research_agent/             # core package — pip installable
│   ├── __init__.py                       # public API: run_research(), ResearchState
│   ├── config.py                         # Pydantic Settings + tier → model mapping
│   ├── llm.py                            # chat model factory per tier (fast/balanced/reasoning)
│   ├── cli.py                            # `paper-research "<topic>" --idea "..."`
│   │
│   ├── core/                             # domain layer — depends on nothing else
│   │   ├── models.py                     # Paper, PaperSource
│   │   ├── state.py                      # ResearchState, ResearchGap (Pydantic)
│   │   ├── errors.py                     # ProviderError, RateLimitError, ...
│   │   └── logging.py
│   │
│   ├── providers/                        # external paper sources
│   │   ├── base.py                       # PaperProvider protocol — add a source = 1 file
│   │   ├── arxiv.py
│   │   └── openalex.py
│   │
│   ├── features/                         # vertical slices — one package per graph node
│   │   ├── planning/                     # plan_queries (LLM planner lands here)
│   │   ├── fetching/                     # fetch_papers: service + dedup + ranking
│   │   ├── contrast/                     # find_gaps (planned)
│   │   ├── novelty/                      # score_novelty (planned)
│   │   └── writing/                      # write_report (planned)
│   │
│   ├── agent/                            # graph assembly only — no business logic
│   │   ├── registry.py                   # ordered NodeSpec list; add a node here
│   │   └── graph.py                      # builds StateGraph from registry, run_research()
│   │
│   └── api/                              # FastAPI layer
│       ├── main.py                       # create_app()
│       ├── routes.py                     # POST /research, GET /health
│       └── schemas.py                    # request/response models
│
└── tests/
    ├── test_fetching.py                  # pure unit tests (dedup, ranking)
    ├── test_graph.py                     # end-to-end graph run with stub provider
    └── integration/test_providers.py     # real APIs — `pytest -m integration`
```

---

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent framework | LangGraph | Stateful multi-node workflow (LLM-agnostic) |
| LLM access | `langchain-openai` → OpenRouter | One OpenAI-compatible client, tier-routed models |
| Data — papers | ArXiv API (`arxiv` PyPI) | Full metadata + abstracts, free |
| Data — citations | OpenAlex API | Citation graph, metadata, citation counts |
| Schemas & config | Pydantic / pydantic-settings | State, domain models, typed settings |
| API | FastAPI + Uvicorn | Expose agent as REST endpoint |
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
git clone https://github.com/yourusername/paper-research-agent
cd paper-research-agent

uv sync

cp .env.example .env
# Edit .env and fill in your API key
```

### Environment variables

```bash
# Required — key for the OpenAI-compatible LLM endpoint
API_KEY=

# Optional — endpoint base URL (default: https://openrouter.ai/api/v1)
LLM_BASE_URL=

# Optional — force every node onto one tier: "fast" | "balanced" | "reasoning"
MODEL_TIER_OVERRIDE=

# Optional — per-tier model names
FAST_MODEL=deepseek/deepseek-v4-flash
BALANCED_MODEL=deepseek/deepseek-v4-pro
REASONING_MODEL=deepseek/deepseek-v4-pro

# Optional — provider limits/timeouts
ARXIV_MAX_RESULTS=8
OPENALEX_MAX_RESULTS=8
REQUEST_TIMEOUT_SECONDS=20

# Optional — OpenAlex API key (higher rate limits)
OPENALEX_API_KEY=

# Optional — LangSmith tracing
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=paper-research-agent
```

### Run

```bash
# CLI
uv run paper-research "retrieval augmented generation" --idea "hierarchical chunking for RAG recall"

# REST API
uv run uvicorn paper_research_agent.api.main:app --reload --port 8000
```

### Test

```bash
uv run pytest                   # unit tests (no network)
uv run pytest -m integration    # hits real ArXiv/OpenAlex APIs
uv run ruff check src tests     # lint
```

---

## API

### `POST /research`

Run the research pipeline on a topic or idea.

**Request**
```json
{
  "topic": "retrieval-augmented generation for long documents",
  "user_idea": "using hierarchical chunking to improve RAG recall"
}
```

**Response**
```json
{
  "topic": "retrieval-augmented generation for long documents",
  "papers_found": 16,
  "papers": [
    {
      "title": "…",
      "authors": ["…"],
      "year": 2024,
      "abstract": "…",
      "url": "https://arxiv.org/abs/…",
      "source": "arxiv",
      "citation_count": null
    }
  ],
  "gaps": [],
  "novelty_score": null,
  "novelty_reasoning": null,
  "overlapping_papers": [],
  "report_markdown": null
}
```

> `gaps`, `novelty_*`, and `report_markdown` are populated once the contrast, novelty, and writer nodes land — the schema is final, the values are not yet computed.

### `GET /health`

```json
{ "status": "ok" }
```

---

## Design principles

Applied now:

| Principle | Where |
|---|---|
| **Schema-first state** — Pydantic models define the contract before any prompt exists | `core/state.py`, `core/models.py` |
| **Tier routing, not model names** — nodes request fast/balanced/reasoning; mapping lives in one place | `config.py`, `llm.py` |
| **Provider protocol** — adding a paper source is one file implementing `PaperProvider` | `providers/base.py` |
| **Extraction vs calculation** — dedup and ranking are pure Python, not LLM calls | `features/fetching/` |
| **Helpful errors as prompts** — rate limits raise `"Wait or set OPENALEX_API_KEY"`, not a bare 429 | `providers/openalex.py` |
| **Graph = wiring only** — business logic lives in features; the registry is the single extension point | `agent/registry.py` |
| **Tests without network by default** — graph runs end-to-end against a stub provider; real-API tests are opt-in | `tests/` |

Planned alongside the LLM nodes: agentic fetch loop with explicit stop criteria (`max_steps`, timeout), structured output via native JSON Schema mode, quote-first prompting for the contrast node, stable prompt prefixes for KV caching, and an MCP server exposing the paper-search tools (a `tools/` package will reappear then — LLM-facing wrappers over `providers/` with prompt-friendly errors and truncated outputs).

---

## Known limitations

- **Abstract-only**: the agent reads abstracts, not full paper text. Deep methodological gaps may be missed.
- **English only**: ArXiv and OpenAlex coverage is strongest for English-language CS papers.
- **Rate limits**: OpenAlex throttles anonymous usage. For heavy use, set `OPENALEX_API_KEY`.
- **Gap/novelty stages not implemented yet** — current output is the ranked, deduplicated paper list.

---

## Roadmap

- [ ] LLM planner — multi-angle query expansion (fast tier)
- [ ] Contrast node — gap analysis across papers (reasoning tier)
- [ ] Novelty node — score user idea against found gaps (reasoning tier)
- [ ] Writer node — markdown report with citations (balanced tier)
- [ ] Evaluation harness — golden set + operational metrics (`tool_call_count`, token usage, runtime)
- [ ] MCP server (FastMCP) exposing `arxiv_search` / `openalex_search` tools
- [ ] Streamlit demo frontend
- [ ] PDF upload — read your own draft, agent finds gaps in your contribution
- [ ] Semantic search — embedding-based retrieval instead of keyword search
- [ ] Citation graph visualization

---

## Built for

AIE1 Final Project — Module 1: Building a complete AI Agent

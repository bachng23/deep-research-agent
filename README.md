# paper-research-agent

An AI agent that researches academic literature (ArXiv + OpenAlex), reads paper full text, identifies research gaps, surfaces conflicting evidence, and scores the novelty of your idea — built with LangGraph, model-agnostic via any OpenAI-compatible endpoint (OpenRouter by default).

Python 3.12+ · LangGraph · FastAPI · FastMCP · Qdrant · 143 tests

---

## Quickstart

```bash
git clone https://github.com/bachng23/deep-re-search-agent
cd deep-re-search-agent
uv sync

cp .env.example .env        # then set API_KEY (OpenRouter or any OpenAI-compatible endpoint)

uv run paper-research       # terminal UI
```

Type a topic to start a research run, or a question to query papers already read. `ctrl+t` switches mode.

---

## Contents

- [What it does](#what-it-does) · [Pipeline](#pipeline) · [Model routing](#model-routing)
- [Interfaces](#interfaces) — TUI, REST, MCP, Python
- [Evaluation](#evaluation) · [Project structure](#project-structure) · [Design principles](#design-principles)
- [Known limitations](#known-limitations) · [Roadmap](#roadmap)

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

The eval harness lives in [`src/paper_research_agent/eval/`](src/paper_research_agent/eval/): a 13-topic golden set with recorded provenance (survey title, arXiv id, and the exact section each expected gap keyword came from), metrics over a finished run (`grounded_in_fulltext`, `gap_keyword_recall`, `paper_recall`, `citation_coverage`, plus operational counters), heuristic failure analysis, and a runner that repeats each topic k times and reports mean ± std.

```bash
# smoke: 1 topic, 1 run — for iterating without burning credits
uv run python -m paper_research_agent.eval.runner --smoke

# full: every topic, 3 runs each, results under eval/results/<timestamp>/
uv run python -m paper_research_agent.eval.runner --golden all -k 3 --out eval/results
```

Each run directory holds `meta.json` (model ids per tier, temperature, git SHA, package version, date), `runs.json` (per-run metrics), `summary.json` (mean ± std, broken down by ground-truth strength and difficulty), and `states/` (the full `ResearchState` of every run, so any number can be traced to the run that produced it).

### Results

> 🚧 **To be updated.** A full run is in progress; this table stays empty until every figure in it comes from a committed run under `eval/results/`. Nothing here is estimated or filled in by hand.

| Metric | Mean ± std | n |
|---|---|---|
| `grounded_in_fulltext` | _pending_ | — |
| `citation_coverage` | _pending_ | — |
| `gap_keyword_recall` | _pending_ | — |
| `paper_recall` | _pending_ | — |
| `gaps` per run | _pending_ | — |
| `papers` per run | _pending_ | — |

Method, precise metric definitions, golden-set provenance, the failure taxonomy and the known limitations are documented now, ahead of the numbers, in **[`eval/README.md`](eval/README.md)** — so the method can be judged independently of how the results turn out.

**How the harness is kept honest**

- **k = 3 runs per topic.** LLM output is non-deterministic; a single run is not a result, so mean ± sample std is reported with a per-metric `n`.
- **Ground truth comes from published surveys**, never from this agent. Every arXiv id is verified; each topic records which section its expected gap keywords came from, and the 5 topics whose labels rest only on an abstract are flagged as weaker.
- **Unmeasurable is not zero.** A metric that cannot be computed on a run returns `None` and is dropped from the mean and from `n`, rather than counted as 0 or as a perfect score.
- **Memory is off during eval**, or runs 2 and 3 of a topic would read what run 1 cached and collapse the measured variance into an artefact.
- **Two metrics were repaired before baselining.** `citation_coverage` previously returned 1.0 unconditionally (a character-class regex over a body never split from the reference list); `grounded_in_fulltext` rejected correct quotes whose source wrapped across lines. Both fixes raise the numbers relative to older code, so results are not comparable to anything produced before commit `bfa8f12`.

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

cp .env.example .env
# then fill in API_KEY
```

Environment variables (`.env.example` has the full list):

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
- **Eval results not published yet** — the harness is in place and documented; the results table is pending a completed run.
- **Retrieval ranking favours citation count**, so a broadly-cited survey can outrank a more precisely relevant recent paper, and an ambiguous query term can pull in an unrelated field.
- **Rate limits**: OpenAlex throttles anonymous usage. Set `OPENALEX_API_KEY` for heavy use.
- **Semantic result cache can serve a stale run** — a topic within `RESULT_CACHE_TTL_DAYS` and above the similarity threshold short-circuits the graph entirely.

---

## Roadmap

- [x] Eval harness — 13-topic golden set with provenance, k-run variance, failure taxonomy, persisted raw state ([`eval/README.md`](eval/README.md))
- [ ] Publish the results table from a completed run
- [ ] Model-tier ablation — measure the cost/quality tradeoff of routing every node to the `fast` tier
- [ ] Token/cost accounting per run
- [ ] PDF upload — read your own draft, agent finds gaps in your contribution
- [ ] Citation graph traversal via OpenAlex references
- [ ] Web demo frontend

---

## Built for

AIE1 Final Project — Module 1: Building a complete AI Agent

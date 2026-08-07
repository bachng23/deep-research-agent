# Evaluation

> **Status: run in progress.** The results table below is empty on purpose. No
> number appears here until it comes from a committed run under
> `eval/results/`. This file documents what is measured and how, so the method
> can be judged before the numbers exist.

## What is measured

Each run is one full research pass over one topic: plan → fetch → find gaps →
judge coverage → (loop) → read full text → refine gaps → conflicts → novelty →
report. Metrics are computed from the finished `ResearchState`.

| Metric | Definition, precisely |
|---|---|
| `papers` | Number of deduplicated papers in final state. |
| `gaps` | Number of research gaps in final state. |
| `conflicts` | Number of detected disagreements between papers. |
| `rounds` | Research-loop iterations actually executed. |
| `tool_calls` | Successful provider search calls made by the fetch tool-agent. |
| `provider_errors` | Count of recorded error strings (provider, LLM, or reading). |
| `elapsed_s` | Wall-clock seconds for the run. |
| `novelty` | **Self-reported.** The model's own 0–100 score for the user idea. Its mean is not evidence of quality; its spread across repeats is a measure of stability. |
| `grounded_in_fulltext` | Of the gaps that carry an evidence quote, the fraction whose quote appears in some paper's full-text excerpt. Whitespace is collapsed on both sides before matching; wording is compared exactly, so layout differences are forgiven and paraphrase is not. `None` when no excerpt was read or no gap carried a quote. |
| `citation_coverage` | Of the distinct `[n]` markers in the report body (references section excluded), the fraction resolving to a numbered reference. `None` when the report cites nothing inline. |
| `gap_keyword_recall` | Fraction of a topic's expected gap keywords occurring as a case-insensitive substring anywhere in the gap descriptions. **Lexical, not semantic** — it credits a keyword appearing in any context, including one where the agent asserts the opposite. Read as coarse topic overlap, not correctness. |
| `paper_recall` | Fraction of a topic's expected papers whose title fragment appears as a substring of some retrieved title. Also lexical. |
| `meets_min_gaps` | Whether the run produced at least the case's `min_gaps`. |

`None` means *not measurable on this run*. It is dropped from means and from
`n`, never counted as 0 or as 1. Every table reports per-metric `n` for this
reason.

## Golden set

13 topics, in [`../src/paper_research_agent/eval/data/golden_set.json`](../src/paper_research_agent/eval/data/golden_set.json).
Ground truth is derived from published literature, never from this agent's
output. Every arXiv id was verified by fetching its abstract page and matching
the title (2026-08-07).

Label strength is recorded per topic, because it is not uniform:

| `gap_source` | Topics | What the labels are |
|---|---|---|
| `section` | 4 | The survey's own future-work section was retrieved; keywords are the directions it enumerates. Strongest. |
| `debate` | 4 | Contested topics; ground truth is the pair of papers that disagree. Used to exercise conflict detection. |
| `abstract` | 5 | Only the abstract could be retrieved; keywords are the survey's stated scope. **Weakest** — flagged in the data file. |

Spread: 6 narrow / 7 broad. Domains are mostly ML/NLP, with federated
learning, continual learning and time-series forecasting included to reduce
the monoculture.

## Method

- **k = 3 runs per topic.** LLM output is non-deterministic, so a single run is
  not a result. Mean ± sample standard deviation is reported, never a lone
  figure.
- **What is fixed:** `temperature=0.0` at every call site. Model ids per tier,
  base URL, and all retrieval limits are recorded per run.
- **What is not fixed:** no seed. OpenRouter does not guarantee seeded
  determinism end to end, so runs are repeated rather than seeded. This is
  recorded as `seed: null` rather than omitted.
- **Memory is off.** With it on, runs 2 and 3 of a topic would read the cache
  or recall gaps written by run 1, which would collapse the measured variance
  into an artefact.
- **The `idea` field is not passed** in the main arm. The golden labels are
  topic-level, taken from surveys; passing a case's narrow idea makes the
  planner chase the idea instead of the topic and scores the agent against a
  question the labels do not ask. A separate smaller arm passes ideas, to
  exercise novelty scoring.
- **Hard deadline** of 720s per run, enforced with `SIGALRM`. Client-side
  timeouts do not interrupt a socket that stays open and silent.

## Results

_Pending the committed run. This section will contain: the metric table
(mean ± std, with n), the breakdown by label strength and by difficulty, and
the failure distribution._

## Failure taxonomy

Findings are categorised so the distribution can be counted, not just read:

`retrieval_miss` · `reading_failure` · `ungrounded_gap` ·
`citation_resolution` · `thin_output` · `missed_conflict` · `provider_error` ·
`timeout` · `query_repetition` · `novelty_missing`

Both counts are reported: findings per category (one run can fail several
ways) and runs affected per category (the per-run view). Reporting only the
first overstates how many runs are broken.

## Known limitations

Stated plainly, because a reader who finds these unaided will discount
everything else:

1. **13 topics is a small set.** Per-topic means rest on 3 runs. Treat
   differences between topics as indicative, not significant.
2. **Label strength is uneven.** Only 4 of 13 topics have labels drawn from a
   survey's enumerated future-work section. 5 rest on abstracts, which describe
   a survey's *scope* rather than its *open problems* — a topic can score badly
   there while the agent behaves well. Results are broken down by label
   strength for this reason.
3. **Recall metrics are lexical.** Substring matching credits a keyword used in
   any context and misses a correct gap phrased differently. It measures topic
   overlap, not correctness.
4. **Provenance bias.** Topics were chosen partly because a good survey exists,
   which favours mature, well-documented areas. Performance on a genuinely
   novel or poorly surveyed topic is not measured here.
5. **Domain bias.** Mostly ML/NLP. Non-English and non-CS literature is not
   represented.
6. **`novelty` is self-reported** and is not evidence of output quality.
7. **The metrics were repaired before this baseline.** `citation_coverage`
   previously returned 1.0 unconditionally, and `grounded_in_fulltext`
   penalised correct quotes whose source wrapped across lines. Both fixes raise
   the numbers relative to the old code. The contrast prompt was also corrected
   to stop instructing the model to quote the abstract while the metric scored
   against full-text excerpts. Numbers here are therefore not comparable to any
   produced before commit `bfa8f12`.
8. **A single passing run proves little about the tail.** The smoke run
   recorded zero provider errors; the first full run then stalled for 68
   minutes on a silent socket.

## Reproducing

```bash
cp .env.example .env   # then set API_KEY
uv run python -m paper_research_agent.eval.runner --golden all -k 3 --no-idea
```

Each run writes `eval/results/<timestamp>[-tag]/`:

| File | Contents |
|---|---|
| `meta.json` | model id per tier, temperature, seed, base URL, git SHA + dirty flag, package version, python/platform, full eval config |
| `runs.json` | per-run metrics and categorised findings |
| `summary.json` | mean/std/n per metric, breakdowns, failure distribution |
| `states/` | the complete `ResearchState` of every run |

`states/` is what makes the numbers checkable: any figure can be recomputed
from the persisted state without paying for the run again, and
`--resume-from <dir>/states` rescores saved runs with the current metric code.

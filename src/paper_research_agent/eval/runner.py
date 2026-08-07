from __future__ import annotations

import argparse
import json
import statistics
import time

from rich.console import Console
from rich.table import Table

from paper_research_agent.agent.graph import run_research
from paper_research_agent.eval.failure import analyze
from paper_research_agent.eval.golden import GOLDEN, GoldenCase
from paper_research_agent.eval.metrics import score_case

console = Console()


def run_one(
    case: GoldenCase, *, max_iterations: int, timeout_seconds: float | None = None
) -> dict:
    t0 = time.monotonic()
    state = run_research(
        case.topic,
        case.idea,
        read_full_text=True,
        max_iterations=max_iterations,
        timeout_seconds=timeout_seconds,
    )
    row = {"topic": case.topic, "elapsed_s": round(time.monotonic() - t0)}
    row.update(score_case(state, case))
    row["findings"] = analyze(state, case)
    return row


def evaluate(
    cases: list[GoldenCase],
    *,
    repeats: int,
    max_iterations: int,
    timeout_seconds: float | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for case in cases:
        for r in range(repeats):
            console.log(f"[{case.topic[:45]}] run {r + 1}/{repeats}")
            row = run_one(
                case, max_iterations=max_iterations, timeout_seconds=timeout_seconds
            )
            row["run"] = r
            rows.append(row)
    return rows


def _results_table(rows: list[dict]) -> Table:
    cols = [
        "topic", "elapsed_s", "papers", "gaps", "conflicts", "rounds",
        "novelty", "grounded_in_fulltext", "citation_coverage",
        "gap_keyword_recall", "paper_recall", "provider_errors",
    ]
    t = Table(title="eval runs")
    t.add_column("topic", width=22, overflow="ellipsis")  # pin narrow so numbers fit
    for c in cols[1:]:
        t.add_column(c.replace("_", " "), justify="right")
    for r in rows:
        t.add_row(*[str(r.get(c, "")) for c in cols])
    return t


def _variance_table(rows: list[dict]) -> Table:
    "Per-topic mean/std of the high-variance metrics (the report's key concern)."
    by_topic: dict[str, list[dict]] = {}
    for r in rows:
        by_topic.setdefault(r["topic"], []).append(r)

    t = Table(title="variance across repeats")
    t.add_column("topic", overflow="fold")
    for m in ("novelty", "gaps", "grounded_in_fulltext"):
        t.add_column(f"{m} mean")
        t.add_column(f"{m} std")

    for topic, runs in by_topic.items():
        cells = [topic[:45]]
        for m in ("novelty", "gaps", "grounded_in_fulltext"):
            vals = [r[m] for r in runs if r.get(m) is not None]
            if vals:
                cells.append(f"{statistics.mean(vals):.1f}")
                cells.append(f"{statistics.pstdev(vals):.1f}" if len(vals) > 1 else "—")
            else:
                cells += ["—", "—"]
        t.add_row(*cells)
    return t


def _print_failures(rows: list[dict]) -> None:
    console.rule("failure analysis")
    any_findings = False
    for r in rows:
        if r.get("findings"):
            any_findings = True
            console.print(f"[bold]{r['topic'][:70]}[/]  [dim](run {r.get('run', 0)})[/]")
            for f in r["findings"]:
                console.print(f"  [red]•[/] {f}")
    if not any_findings:
        console.print("[green]no findings — all cases within thresholds[/]")


def main() -> None:
    ap = argparse.ArgumentParser(prog="paper-research-eval")
    ap.add_argument("--repeats", type=int, default=1, help="runs per case (variance)")
    ap.add_argument("--max-iterations", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="only first N golden cases")
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="per-run budget in seconds; checked between rounds, not mid-call",
    )
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    cases = GOLDEN[: args.limit] if args.limit else GOLDEN
    rows = evaluate(
        cases,
        repeats=args.repeats,
        max_iterations=args.max_iterations,
        timeout_seconds=args.timeout,
    )

    console.print(_results_table(rows))
    if args.repeats > 1:
        console.print(_variance_table(rows))
    _print_failures(rows)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    console.print(f"[dim]wrote {len(rows)} rows → {args.out}[/]")


if __name__ == "__main__":
    main()

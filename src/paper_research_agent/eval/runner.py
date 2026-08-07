from __future__ import annotations

import argparse
import json
import platform
import re
import signal
import statistics
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rich.console import Console
from rich.table import Table

from paper_research_agent.agent.graph import run_research
from paper_research_agent.config import get_settings
from paper_research_agent.core.state import ResearchState
from paper_research_agent.eval.failure import analyze, distribution, runs_affected
from paper_research_agent.eval.golden import GOLDEN, GoldenCase
from paper_research_agent.eval.metrics import score_case

console = Console()

# Metrics aggregated as mean +/- std. Everything else in a row is either an
# identifier or a boolean and is summarised separately.
NUMERIC_METRICS = (
    "papers",
    "gaps",
    "conflicts",
    "rounds",
    "tool_calls",
    "provider_errors",
    "elapsed_s",
    "novelty",
    "grounded_in_fulltext",
    "citation_coverage",
    "gap_keyword_recall",
    "paper_recall",
)

# Reported separately from the agent's own quality: novelty is the model
# scoring itself, so its mean is not evidence of anything. Its spread across
# repeats is, which is why it stays in NUMERIC_METRICS.
SELF_REPORTED = ("novelty",)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
    except Exception:
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
    except Exception:
        return None
    return bool(out.stdout.strip()) if out.returncode == 0 else None


def _package_version() -> str | None:
    try:
        return version("paper-research-agent")
    except PackageNotFoundError:
        return None


def run_metadata(
    *,
    repeats: int,
    max_iterations: int,
    n_cases: int,
    with_idea: bool = True,
    deadline_seconds: float | None = None,
) -> dict:
    """Everything needed to say what produced a number.

    A result without this is not reproducible, and a reader is right to
    discount it.
    """
    s = get_settings()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "package_version": _package_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "models": {
            "fast": s.fast_model,
            "balanced": s.balanced_model,
            "reasoning": s.reasoning_model,
            "embedding": s.embedding_model,
            "tier_override": s.model_tier_override,
        },
        "llm": {
            # Fixed across every node; no call site overrides it.
            "temperature": 0.0,
            # Not set: OpenRouter does not guarantee seeded determinism
            # end-to-end, so runs are repeated instead of seeded.
            "seed": None,
            "base_url": s.llm_base_url,
            "timeout_seconds": s.llm_timeout_seconds,
        },
        "config": {
            "read_max_papers": s.read_max_papers,
            "arxiv_max_results": s.arxiv_max_results,
            "openalex_max_results": s.openalex_max_results,
            "max_new_papers_per_round": s.max_new_papers_per_round,
            "fulltext_chunk_chars": s.fulltext_chunk_chars,
            "fulltext_top_k": s.fulltext_top_k,
        },
        "eval": {
            "cases": n_cases,
            "repeats": repeats,
            "max_iterations": max_iterations,
            "read_full_text": True,
            "use_memory": False,
            # The golden labels are topic-level, taken from surveys. Passing a
            # narrow idea makes the planner chase the idea instead of the
            # topic, so recall would score a question the labels do not ask.
            "with_idea": with_idea,
            "deadline_seconds": deadline_seconds,
        },
    }


@contextmanager
def hard_deadline(seconds: float | None):
    """Abort a run that blows its wall-clock budget.

    state.timeout_seconds is only checked between rounds, and client-side
    timeouts only cover calls that fail cleanly. Neither stops a socket that
    stays open and silent, which is what stalled a full run for 68 minutes.
    SIGALRM interrupts the interpreter wherever it is waiting.
    """
    if not seconds or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _fire(_signum, _frame):
        raise TimeoutError(f"run exceeded hard deadline of {seconds:.0f}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _slug(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:60]


def run_one(
    case: GoldenCase,
    *,
    max_iterations: int,
    timeout_seconds: float | None = None,
    with_idea: bool = True,
    deadline_seconds: float | None = None,
) -> tuple[dict, ResearchState]:
    t0 = time.monotonic()
    with hard_deadline(deadline_seconds):
        state = run_research(
            case.topic,
            case.idea if with_idea else None,
            read_full_text=True,
            max_iterations=max_iterations,
            # Memory off: a cached or recalled result would make run 2 and 3 of
            # the same topic depend on run 1, destroying the variance measure.
            use_memory=False,
            timeout_seconds=timeout_seconds,
        )
    row = _score(state, case)
    row["elapsed_s"] = round(time.monotonic() - t0)
    return row, state


def evaluate(
    cases: list[GoldenCase],
    *,
    repeats: int,
    max_iterations: int,
    timeout_seconds: float | None = None,
    states_dir: Path | None = None,
    with_idea: bool = True,
    deadline_seconds: float | None = None,
    resume_from: Path | None = None,
) -> list[dict]:
    rows: list[dict] = []
    total = len(cases) * repeats
    done = 0

    for case in cases:
        for r in range(repeats):
            done += 1
            console.log(f"({done}/{total}) [{case.topic[:45]}] run {r + 1}/{repeats}")

            cached = _reuse(resume_from, case, r)
            if cached is not None:
                console.log("  [dim]reusing saved state[/]")
                row = _score(cached, case)
                row["run"] = r
                row["reused"] = True
                rows.append(row)
                if states_dir is not None:
                    _write_state(states_dir, case, r, cached)
                continue

            try:
                row, state = run_one(
                    case,
                    max_iterations=max_iterations,
                    timeout_seconds=timeout_seconds,
                    with_idea=with_idea,
                    deadline_seconds=deadline_seconds,
                )
            except Exception as e:
                # A crashed run is a data point, not a reason to lose the ones
                # already paid for.
                console.log(f"[red]run crashed: {e}[/]")
                rows.append(
                    {"topic": case.topic, "run": r, "crashed": repr(e), "findings": []}
                )
                continue

            row["run"] = r
            rows.append(row)

            if states_dir is not None:
                _write_state(states_dir, case, r, state)
    return rows


def _state_path(directory: Path, case: GoldenCase, run: int) -> Path:
    return directory / f"{_slug(case.topic)}--run{run}.json"


def _write_state(directory: Path, case: GoldenCase, run: int, state: ResearchState) -> None:
    _state_path(directory, case, run).write_text(
        state.model_dump_json(indent=2, exclude={"started_at"}), encoding="utf-8"
    )


def _reuse(resume_from: Path | None, case: GoldenCase, run: int) -> ResearchState | None:
    """A previously persisted run of this exact case, if one exists.

    Metrics are recomputed from the state rather than copied, so a reused run
    is scored by the current metric code -- otherwise a resumed eval would mix
    two definitions of the same number.
    """
    if resume_from is None:
        return None
    path = _state_path(resume_from, case, run)
    if not path.exists():
        return None
    try:
        return ResearchState.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _score(state: ResearchState, case: GoldenCase) -> dict:
    row: dict = {"topic": case.topic, "elapsed_s": None}
    row.update(score_case(state, case))
    row["difficulty"] = case.difficulty
    row["gap_source"] = case.provenance.granularity if case.provenance else "unknown"
    row["findings"] = [f.model_dump() for f in analyze(state, case)]
    return row


# --- aggregation ---------------------------------------------------------------


def aggregate(rows: list[dict]) -> dict[str, dict]:
    """Mean/std per metric over all runs.

    None means "not measurable on that run" and is dropped from both the mean
    and the count -- never coerced to zero. `n` is therefore per-metric, and is
    reported so a mean over few points is visible as such.
    """
    out: dict[str, dict] = {}
    for metric in NUMERIC_METRICS:
        vals = [
            r[metric] for r in rows if r.get(metric) is not None and "crashed" not in r
        ]
        if not vals:
            out[metric] = {"mean": None, "std": None, "n": 0}
            continue
        out[metric] = {
            "mean": round(statistics.mean(vals), 3),
            # Sample std: these runs are a sample of the agent's behaviour, not
            # the population of all runs.
            "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else None,
            "n": len(vals),
        }
    return out


def aggregate_by(rows: list[dict], key: str) -> dict[str, dict[str, dict]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(r.get(key, "unknown")), []).append(r)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def per_topic_variance(rows: list[dict]) -> dict[str, dict]:
    "Spread across repeats of the same topic -- the non-determinism measurement."
    by_topic: dict[str, list[dict]] = {}
    for r in rows:
        if "crashed" not in r:
            by_topic.setdefault(r["topic"], []).append(r)
    return {topic: aggregate(runs) for topic, runs in by_topic.items()}


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if "crashed" not in r]
    return {
        "runs_total": len(rows),
        "runs_crashed": len(rows) - len(ok),
        "meets_min_gaps": sum(1 for r in ok if r.get("meets_min_gaps")),
        "runs_with_findings": sum(1 for r in ok if r.get("findings")),
        "failure_distribution": distribution(ok),
        "failure_runs_affected": runs_affected(ok),
        "overall": aggregate(rows),
        "by_gap_source": aggregate_by(ok, "gap_source"),
        "by_difficulty": aggregate_by(ok, "difficulty"),
        "per_topic": per_topic_variance(rows),
    }


# --- output --------------------------------------------------------------------


def _fmt(stat: dict) -> str:
    if stat["mean"] is None:
        return "—"
    if stat["std"] is None:
        return f"{stat['mean']:.2f}"
    return f"{stat['mean']:.2f} ± {stat['std']:.2f}"


def _summary_table(summary: dict) -> Table:
    t = Table(title="metrics (mean ± std over all runs)")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_column("n", justify="right")
    for metric in NUMERIC_METRICS:
        stat = summary["overall"][metric]
        label = f"{metric} (self-reported)" if metric in SELF_REPORTED else metric
        t.add_row(label, _fmt(stat), str(stat["n"]))
    return t


def _breakdown_table(summary: dict, key: str, title: str) -> Table:
    t = Table(title=title)
    t.add_column(key)
    metrics = ("gap_keyword_recall", "paper_recall", "grounded_in_fulltext", "gaps")
    for m in metrics:
        t.add_column(m.replace("_", " "), justify="right")
    for group, stats in summary[key].items():
        t.add_row(group, *[_fmt(stats[m]) for m in metrics])
    return t


def _failure_table(summary: dict, n_runs: int) -> Table:
    t = Table(title="failure distribution")
    t.add_column("category")
    t.add_column("findings", justify="right")
    t.add_column("runs hit", justify="right")
    t.add_column("% of runs", justify="right")
    affected = summary["failure_runs_affected"]
    for category, count in summary["failure_distribution"].items():
        hit = affected.get(category, 0)
        share = f"{hit / n_runs:.0%}" if n_runs else "—"
        t.add_row(category, str(count), str(hit), share)
    return t


def _print_failures(rows: list[dict]) -> None:
    console.rule("findings per run")
    any_findings = False
    for r in rows:
        if r.get("findings") or r.get("crashed"):
            any_findings = True
            console.print(f"[bold]{r['topic'][:70]}[/]  [dim](run {r.get('run', 0)})[/]")
            if r.get("crashed"):
                console.print(f"  [red]•[/] crashed: {r['crashed']}")
            for f in r.get("findings", []):
                console.print(f"  [red]•[/] [yellow]{f['category']}[/] {f['detail'][:110]}")
    if not any_findings:
        console.print("[green]no findings — all cases within thresholds[/]")


def main() -> None:
    ap = argparse.ArgumentParser(prog="paper-research-eval")
    ap.add_argument(
        "--golden", default="all", help="'all' or a substring matching topics to run"
    )
    ap.add_argument("-k", "--repeats", type=int, default=3, help="runs per case")
    ap.add_argument("--max-iterations", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="only first N cases")
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="per-run budget in seconds; checked between rounds, not mid-call",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="1 case, 1 run — for iterating without burning credits",
    )
    ap.add_argument(
        "--no-idea",
        action="store_true",
        help="research the topic only; the golden labels are topic-level",
    )
    ap.add_argument(
        "--deadline",
        type=float,
        default=720.0,
        help="hard wall-clock cap per run in seconds (0 disables)",
    )
    ap.add_argument(
        "--resume-from",
        default=None,
        help="a previous states/ directory; matching runs are rescored, not re-run",
    )
    ap.add_argument("--tag", default=None, help="label for this run directory")
    ap.add_argument("--out", default="eval/results", help="results root directory")
    args = ap.parse_args()

    cases = list(GOLDEN)
    if args.golden != "all":
        needle = args.golden.lower()
        cases = [c for c in cases if needle in c.topic.lower()]
    if args.limit:
        cases = cases[: args.limit]

    repeats = args.repeats
    if args.smoke:
        cases, repeats = cases[:1], 1

    if not cases:
        console.print(f"[red]no golden case matches {args.golden!r}[/]")
        raise SystemExit(1)

    with_idea = not args.no_idea
    meta = run_metadata(
        repeats=repeats,
        max_iterations=args.max_iterations,
        n_cases=len(cases),
        with_idea=with_idea,
        deadline_seconds=args.deadline or None,
    )
    stamp = meta["timestamp_utc"].replace(":", "").replace("-", "")
    run_dir = Path(args.out) / (f"{stamp}-{args.tag}" if args.tag else stamp)
    states_dir = run_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[dim]models: {meta['models']}[/]")
    console.print(f"[dim]commit: {meta['git_sha']} (dirty={meta['git_dirty']})[/]")
    console.print(f"[dim]writing to {run_dir}[/]")
    if meta["git_dirty"]:
        console.print("[yellow]working tree is dirty — this run is not reproducible from the recorded SHA[/]")

    rows = evaluate(
        cases,
        repeats=repeats,
        max_iterations=args.max_iterations,
        timeout_seconds=args.timeout,
        states_dir=states_dir,
        with_idea=with_idea,
        deadline_seconds=args.deadline or None,
        resume_from=Path(args.resume_from) if args.resume_from else None,
    )
    summary = summarise(rows)

    console.print(_summary_table(summary))
    console.print(_breakdown_table(summary, "by_gap_source", "by ground-truth strength"))
    console.print(_breakdown_table(summary, "by_difficulty", "by topic difficulty"))
    console.print(_failure_table(summary, summary["runs_total"] - summary["runs_crashed"]))
    _print_failures(rows)

    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "runs.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(f"[dim]wrote {len(rows)} rows + {len(cases) * repeats} states → {run_dir}[/]")


if __name__ == "__main__":
    main()

from __future__ import annotations

import pyfiglet
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Input, Static

from paper_research_agent.agent.graph import stream_research
from paper_research_agent.config import get_settings

# friendly labels for the graph nodes as they stream past
_NODE_LABELS = {
    "plan_queries": "planning queries",
    "fetch_papers": "fetching papers",
    "find_gaps": "finding gaps",
    "assess_coverage": "assessing coverage",
    "judge_coverage": "judging coverage",
    "read_papers": "reading full texts",
    "refine_gaps": "refining gaps with full text",
    "rank_gaps": "ranking gaps",
    "find_conflicts": "finding conflicts",
    "score_novelty": "scoring novelty",
    "write_report": "writing report",
}

_MAX_REFERENCES_SHOWN = 5
_EXIT = {"exit", "quit", ":q"}

_ASCII = pyfiglet.figlet_format("re-search", font="ansi_shadow").rstrip("\n")


def _model_lines() -> list[Text]:
    "LLM models in use (grouped by tier), read from config, as a list of lines."
    try:
        s = get_settings()
        provider = "OpenRouter" if "openrouter" in s.llm_base_url else s.llm_base_url
        grouped: dict[str, list[str]] = {}
        for tier, model in (
            ("reasoning", s.reasoning_model),
            ("balanced", s.balanced_model),
            ("fast", s.fast_model),
        ):
            grouped.setdefault(model, []).append(tier)

        lines = [Text(f"model · {provider}", style="bold")]
        for model, tiers in grouped.items():
            line = Text("  ")
            line.append(f"{'/'.join(tiers)}: ", style="dim")
            line.append(model, style="cyan")
            lines.append(line)
        return lines
    except Exception:
        return [Text("model · unknown", style="dim")]


def _welcome() -> Panel:
    art = Text(_ASCII, style="bold cyan", no_wrap=True)
    art_h = _ASCII.count("\n") + 1

    # right side: tips, then the model block below (kept short to avoid wrapping)
    right: list[Text] = [Text("Tips for getting started", style="bold")]
    for tip in (
        "1. Type a research topic and press Enter.",
        "2. Searches arXiv + OpenAlex, reads full text.",
        "3. Finds gaps & conflicts → cited report (top-5).",
        "4. exit / quit / Ctrl-C to leave.",
    ):
        right.append(Text(f"  {tip}", style="dim"))
    right.append(Text(""))  # spacer
    right.extend(_model_lines())

    # a vertical rule separating the ascii (left) from tips+model (right)
    divider = Text("\n".join("│" for _ in range(max(art_h, len(right)))), style="blue")

    grid = Table.grid(padding=(0, 3))
    grid.add_column()  # ascii art
    grid.add_column()  # divider
    grid.add_column()  # tips + model
    grid.add_row(art, divider, Group(*right))

    return Panel(
        grid,
        title="paper · research · agent",
        title_align="left",
        border_style="blue",
        padding=(1, 2),
    )


def _node_count(node: str, state) -> int | None:
    "A small count to show beside a finished step, for a sense of progress."
    if node == "fetch_papers":
        return len(state.papers)
    if node == "read_papers":
        return sum(1 for p in state.papers if p.full_text_excerpt)
    if node in ("find_gaps", "refine_gaps"):
        return len(state.gaps)
    if node == "find_conflicts":
        return len(state.conflicts)
    return None


def _truncate_references(md: str, limit: int = _MAX_REFERENCES_SHOWN) -> str:
    "Display-only: keep the full report but show just the top-N references + '…'."
    marker = "## References"
    i = md.find(marker)
    if i == -1:
        return md

    before, block = md[:i], md[i:].splitlines()
    header, entries = block[0], [line for line in block[1:] if line.strip()]

    shown = [header, *entries[:limit]]
    if len(entries) > limit:
        shown.append("…")
    return before + "\n".join(shown)


class ResearchTUI(App):
    TITLE = "Paper Research Agent"
    CSS = """
    Screen { background: ansi_default; }
    VerticalScroll { background: ansi_default; padding: 0 1; }
    Footer { background: ansi_default; }
    #ask {
        dock: bottom;
        height: 3;
        background: ansi_default;
        border: round ansi_blue;
    }
    """

    def on_mount(self) -> None:
        self._busy = False
        self._entries: list[tuple[str, str]] = []  # (kind, text); kind: step|note
        # one persistent Spinner -> its start_time is kept, so it actually animates
        self._spinner = Spinner("dots", text=Text(" working…", style="cyan"))
        self.query_one("#ask", Input).border_title = "ask"
        self.query_one("#log", VerticalScroll).mount(Static(_welcome()))
        self.set_interval(1 / 12, self._tick)
        self.query_one("#ask", Input).focus()

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="log")
        yield Input(id="ask")
        yield Footer()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        topic = event.value.strip()
        if not topic:
            return
        if topic.lower() in _EXIT:
            self.exit()
            return

        event.input.value = ""
        event.input.disabled = True
        log = self.query_one("#log", VerticalScroll)

        await log.mount(
            Static(Text("you ▸ ", style="bold cyan") + Text(topic, style="bold"))
        )
        self._entries = []
        self._plan_seen = 0
        self._steps = Static()
        await log.mount(self._steps)
        self._busy = True
        log.scroll_end()
        self._research(topic)

    def _tick(self) -> None:
        if self._busy:  # re-rendering animates the active step's spinner
            self._steps.update(self._render_steps())

    def _render_steps(self) -> Group:
        last_step = max(
            (i for i, (k, _) in enumerate(self._entries) if k == "step"), default=-1
        )
        rows: list = []
        for i, (kind, text) in enumerate(self._entries):
            if kind == "note":
                rows.append(Text(f"  {text}", style="yellow"))
            elif i == last_step and self._busy:
                rows.append(self._spinner)  # reused -> animates across ticks
            else:
                rows.append(Text(f"  ✓ {text}", style="green"))
        return Group(*rows)

    @work(thread=True)
    def _research(self, topic: str) -> None:
        final = None
        try:
            for event in stream_research(topic, read_full_text=True, max_iterations=2):
                if event[0] == "node":
                    _, node, state = event

                    if node == "plan_queries":
                        self._plan_seen += 1
                        if self._plan_seen > 1:  # looped back: coverage thin
                            self.call_from_thread(
                                self._add,
                                "note",
                                f"↻ coverage still thin — digging deeper "
                                f"(round {self._plan_seen}, "
                                f"{len(state.open_gaps)} open gaps)…",
                            )

                    label = _NODE_LABELS.get(node, str(node))
                    count = _node_count(node, state)
                    suffix = f"  [{count}]" if count is not None else ""
                    self.call_from_thread(self._add, "step", f"{label}{suffix}")
                else:
                    final = event[1]
            self.call_from_thread(self._show_report, final)
        except Exception as e:  # never let a run kill the UI
            self.call_from_thread(self._add, "note", f"error: {e}")
        finally:
            self.call_from_thread(self._finish_run)

    def _add(self, kind: str, text: str) -> None:
        self._entries.append((kind, text))
        if kind == "step":  # move the spinner's label to the new active step
            self._spinner.text = Text(f" {text}", style="cyan")
        self._steps.update(self._render_steps())
        self.query_one("#log", VerticalScroll).scroll_end()

    def _show_report(self, final) -> None:
        log = self.query_one("#log", VerticalScroll)
        if final is None:
            log.mount(Static(Text("No result.", style="red")))
            return

        if final.report_markdown:
            log.mount(Static(Markdown(_truncate_references(final.report_markdown))))
        else:
            log.mount(Static(Text("No report generated.", style="yellow")))

        log.mount(
            Static(
                Panel.fit(
                    f"rounds={final.iteration}   papers={len(final.papers)}   "
                    f"gaps={len(final.gaps)}   conflicts={len(final.conflicts)}   "
                    f"novelty={final.novelty_score}",
                    title="summary",
                    border_style="cyan",
                )
            )
        )
        if final.errors:
            log.mount(
                Static(Text(f"{len(final.errors)} error(s): {final.errors}", style="red"))
            )
        log.scroll_end()

    def _finish_run(self) -> None:
        self._busy = False
        self._steps.update(self._render_steps())  # active step settles to ✓
        inp = self.query_one("#ask", Input)
        inp.disabled = False
        inp.focus()


def main() -> None:
    ResearchTUI(ansi_color=True).run()


if __name__ == "__main__":
    main()

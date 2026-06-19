from __future__ import annotations

import random

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
from paper_research_agent.features.qa import stream_answer

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

# little bits of personality, picked at random for variety
_SAY = {
    "new_research": [
        "Hmm, I don't have anything on this yet — let me go dig.",
        "New territory for me. Off to the literature.",
        "Haven't looked into this one. Let me find out.",
        "Don't know this one yet — give me a moment to research it.",
        "Fresh topic. Pulling up papers now.",
    ],
    "cached": [
        "Oh, I've researched this before — pulling it from memory.",
        "I remember this one. Here's what I found last time.",
        "Seen this already — loading the previous findings.",
        "No need to redo this; I've got it saved.",
    ],
    "recall": [
        "This rings a bell — I'll build on related work I've seen.",
        "I've looked at something close before; using it as a head start.",
        "Related to past research — reusing what I learned there.",
        "I already have some relevant notes; building on them.",
    ],
    "qa_thinking": [
        "Let me check what I've read.",
        "Digging through my notes.",
        "Looking that up in what I've gathered.",
        "One sec — searching my memory.",
    ],
}


def _say(kind: str) -> str:
    return random.choice(_SAY[kind])


_MODE_LABELS = {
    "auto": "auto (router decides)",
    "research": "deep research",
    "qa": "Q&A · memory",
}

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
        "1. Deep research: type a topic → fetch + analyze.",
        "2. Ctrl+T → Q&A over all researched papers.",
        "3. Cited report (top-5 refs).",
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

    BINDINGS = [("ctrl+t", "toggle_mode", "mode: auto/research/Q&A")]

    def on_mount(self) -> None:
        self._busy = False
        self._mode = "auto"  # auto | research | qa
        self._report: Static | None = None
        self._entries: list[tuple[str, str]] = []  # (kind, text); kind: step|note
        # one persistent Spinner -> its start_time is kept, so it actually animates
        self._spinner = Spinner("dots", text=Text(" working…", style="cyan"))

        # persistent cross-session memory: research accumulates, Q&A reads it all
        get_settings().use_memory = True

        self.query_one("#log", VerticalScroll).mount(Static(_welcome()))
        self._update_mode_label()
        self.set_interval(1 / 12, self._tick)
        self.query_one("#ask", Input).focus()

    def action_toggle_mode(self) -> None:
        self._mode = {"auto": "research", "research": "qa", "qa": "auto"}[self._mode]
        self._update_mode_label()

    def _update_mode_label(self) -> None:
        self.query_one("#ask", Input).border_title = (
            f"{_MODE_LABELS[self._mode]}   (ctrl+t to switch)"
        )

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="log")
        yield Input(id="ask")
        yield Footer()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if text.lower() in _EXIT:
            self.exit()
            return

        event.input.value = ""
        event.input.disabled = True
        log = self.query_one("#log", VerticalScroll)
        await log.mount(
            Static(Text("you ▸ ", style="bold cyan") + Text(text, style="bold"))
        )

        if self._mode == "auto":
            self._route(text)               # worker: classify -> _start
        else:
            await self._start(self._mode, text)

    @work(thread=True)
    def _route(self, text: str) -> None:
        "Classify intent off the UI thread, then dispatch on the main thread."
        from paper_research_agent.agent.router import route

        mode = route(text)
        self.call_from_thread(self._start, mode, text)

    async def _start(self, mode: str, text: str) -> None:
        log = self.query_one("#log", VerticalScroll)
        if mode == "qa":
            placeholder = Static(Text(f"⠋ {_say('qa_thinking')}", style="cyan"))
            await log.mount(placeholder)
            log.scroll_end()
            self._answer(text, placeholder)
        else:
            self._entries = []
            self._plan_seen = 0
            self._report = None
            self._steps = Static()
            await log.mount(self._steps)
            self._busy = True
            log.scroll_end()
            self._research(text)

    def _tick(self) -> None:
        if self._busy and hasattr(self, "_steps"):  # animate the active step
            self._steps.update(self._render_steps())

    def _render_steps(self) -> Group:
        last_step = max(
            (i for i, (k, _) in enumerate(self._entries) if k == "step"), default=-1
        )
        rows: list = []
        for i, (kind, text) in enumerate(self._entries):
            if kind == "say":
                rows.append(Text(f"  {text}", style="bold cyan"))
            elif kind == "note":
                rows.append(Text(f"  {text}", style="yellow"))
            elif i == last_step and self._busy:
                rows.append(self._spinner)  # reused -> animates across ticks
            else:
                rows.append(Text(f"  ✓ {text}", style="green"))
        return Group(*rows)

    @work(thread=True)
    def _research(self, topic: str) -> None:
        final = None
        opened = False
        try:
            for event in stream_research(
                topic, read_full_text=True, max_iterations=2, use_memory=True
            ):
                if event[0] == "node":
                    _, node, state = event

                    if not opened:  # first words: new vs. recalled
                        opened = True
                        kind = "recall" if state.recalled_gaps else "new_research"
                        self.call_from_thread(self._add, "say", _say(kind))

                    if node == "plan_queries":
                        self._plan_seen += 1
                        if self._plan_seen > 1:  # new round -> wipe the old round's steps
                            self.call_from_thread(self._new_round, len(state.open_gaps))

                    label = _NODE_LABELS.get(node, str(node))
                    count = _node_count(node, state)
                    suffix = f"  [{count}]" if count is not None else ""
                    self.call_from_thread(self._add, "step", f"{label}{suffix}")
                elif event[0] == "cached":
                    cached = event[1]
                    if not opened:
                        opened = True
                        self.call_from_thread(self._add, "say", _say("cached"))
                    self.call_from_thread(
                        self._add, "note",
                        f"loaded from memory — researched before "
                        f"({len(cached.papers)} papers, <= "
                        f"{get_settings().research_ttl_days}d old)",
                    )
                elif event[0] == "token":
                    self.call_from_thread(self._stream_report, event[1])
                else:
                    final = event[1]
            self.call_from_thread(self._show_report, final)
        except Exception as e:  # never let a run kill the UI
            self.call_from_thread(self._add, "note", f"error: {e}")
        finally:
            self.call_from_thread(self._finish_run)

    @work(thread=True)
    def _answer(self, question: str, placeholder: Static) -> None:
        "Stream a Q&A answer token-by-token into the placeholder (typing effect)."
        acc = ""
        try:
            for token in stream_answer(question):
                acc += token
                self.call_from_thread(placeholder.update, Text(acc))  # plain while typing
        except Exception as e:  # never let a failure kill the UI
            acc = f"error: {e}"
        # final pass: render the whole answer as markdown
        self.call_from_thread(placeholder.update, Markdown(acc) if acc else Text(""))
        self.call_from_thread(self._finish_run)

    def _new_round(self, open_gaps: int) -> None:
        "Start a fresh round: clear the previous round's step lines."
        self._entries = [(
            "note",
            f"↻ coverage still thin — digging deeper "
            f"(round {self._plan_seen}, {open_gaps} open gaps)…",
        )]
        self._steps.update(self._render_steps())

    def _add(self, kind: str, text: str) -> None:
        self._entries.append((kind, text))
        if kind == "step":  # move the spinner's label to the new active step
            self._spinner.text = Text(f" {text}", style="cyan")
        self._steps.update(self._render_steps())
        self.query_one("#log", VerticalScroll).scroll_end()

    def _stream_report(self, token: str) -> None:
        "Type the report out as the writer streams it."
        if self._report is None:  # first token -> mount a live report widget
            self._report = Static()
            self.query_one("#log", VerticalScroll).mount(self._report)
            self._report_text = ""
        self._report_text += token
        self._report.update(Text(self._report_text))  # plain text while typing

    def _show_report(self, final) -> None:
        log = self.query_one("#log", VerticalScroll)
        if final is None:
            log.mount(Static(Text("No result.", style="red")))
            return

        rendered = (
            Markdown(_truncate_references(final.report_markdown))
            if final.report_markdown
            else Text("No report generated.", style="yellow")
        )
        if self._report is not None:  # replace the streamed body with the full render
            self._report.update(rendered)
        else:
            log.mount(Static(rendered))

        log.mount(
            Static(
                Panel.fit(
                    f"rounds={final.iteration}   papers={len(final.papers)}   "
                    f"read={sum(1 for p in final.papers if p.full_text_excerpt)}   "
                    f"gaps={len(final.gaps)}   conflicts={len(final.conflicts)}   "
                    f"novelty={final.novelty_score}",
                    title="summary",
                    border_style="cyan",
                )
            )
        )
        if final.errors:
            log.mount(
                Static(
                    Text(f"{len(final.errors)} error(s): {final.errors}", style="red")
                )
            )
        log.scroll_end()

    def _finish_run(self) -> None:
        self._busy = False
        if hasattr(self, "_steps"):
            self._steps.update(self._render_steps())  # active step settles to ✓
        self.query_one("#log", VerticalScroll).scroll_end()
        inp = self.query_one("#ask", Input)
        inp.disabled = False
        inp.focus()


def main() -> None:
    ResearchTUI(ansi_color=True).run()


if __name__ == "__main__":
    main()

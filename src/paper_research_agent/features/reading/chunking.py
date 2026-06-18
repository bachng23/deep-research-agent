from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FINDINGS_KEYWORDS = (
    "results",
    "discussion",
    "conclusion",
    "evaluation",
    "experiment",
    "finding",
    "limitation",
    "analysis",
)
@dataclass
class Chunk:
    text: str
    section: str | None = None
    is_findings: bool = False


def chunk_text(text: str, *, max_chars: int = 1500, overlap: int = 200) -> list[Chunk]:
    """Split full text into chunks, keeping section metadata.

    Markdown (arXiv HTML) splits by headings; PDF (no headings) falls back to a
    single section windowed by length. `is_findings` flags Results/Discussion/...
    sections where claims & contradictions live."""

    chunks: list[Chunk] = []

    for title, body in _spilt_sections(text):
        is_findings = _is_findings(title)
        for piece in _window(body, max_chars, overlap):
            chunks.append(Chunk(text=piece, section=title, is_findings=is_findings))
    return chunks


def _spilt_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    title: str | None = None
    lines: list[str] = []

    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            if lines:
                sections.append((title, "\n".join(lines).strip()))
            title = m.group(2).strip()
            lines = []
        else:
            lines.append(line)

    if lines:
        sections.append((title, "\n".join(lines).strip()))

    return [(t, b) for t, b in sections if b]


def _is_findings(title: str | None) -> bool:
    if not title:
        return False
    low = title.lower()
    return any(k in low for k in _FINDINGS_KEYWORDS)


def _window(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []

    pieces, start = [], 0
    while start < len(text):
        pieces.append(text[start : start + max_chars])
        start += max_chars - overlap
    return pieces

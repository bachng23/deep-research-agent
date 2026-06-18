from __future__ import annotations

import logging
import re

import lxml.html
import requests
import trafilatura

from paper_research_agent.config import get_settings
from paper_research_agent.core.models import Paper

logger = logging.getLogger(__name__)


_ARXIV_ID_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/(?P<id>[\w.\-]+?)(?:v\d+)?(?:\.pdf)?/?$",
    re.IGNORECASE,
)

_LTX_LEVEL = {
    "ltx_title_abstract": "##",
    "ltx_title_section": "##",
    "ltx_title_subsection": "###",
    "ltx_title_subsubsection": "####",
    "ltx_title_paragraph": "#####",
}

_LTX_DROP = (
    "ltx_bibliography",
    "ltx_appendix",
    "ltx_acknowledgement",
    "ltx_acknowledgements",
)


def fetch_full_text(paper: Paper) -> str | None:
    """Full text map: arXiv HTML -> PDF -> None.
    Any network/parse error degrades to None."""
    arxiv_id = _arxiv_id_from(paper)

    if arxiv_id:
        text = _from_arxiv_html(arxiv_id)
        if text:
            return text

    if paper.pdf_url:
        text = _from_pdf(paper.pdf_url)
        if text:
            return text

    return None


def _arxiv_id_from(paper: Paper) -> str | None:
    for candidate in (paper.url, paper.pdf_url):
        if not candidate:
            continue
        m = _ARXIV_ID_RE.search(candidate)
        if m:
            return m.group("id")
    return None


def _from_arxiv_html(arxiv_id: str) -> str | None:
    url = f"https://arxiv.org/html/{arxiv_id}"
    html = _get(url)
    if not html:
        return None

    text = _parse_arxiv_html(html)
    if text:
        return text

    text = trafilatura.extract(html, output_format="markdown", include_tables=False)

    return text or None


def _from_pdf(pdf_url: str) -> str | None:
    import pymupdf

    content = _get(pdf_url, as_bytes=True)
    if not content:
        return None

    try:
        with pymupdf.open(stream=content, filetype="pdf") as doc:
            return "\n\n".join(page.get_text() for page in doc) or None
    except Exception as e:
        logger.warning("pdf parse failed for %s: %s", pdf_url, e)
        return None


def _get(url: str, *, as_bytes: bool = False) -> str | bytes | None:
    settings = get_settings()

    try:
        resp = requests.get(url, timeout=settings.request_timeout_seconds)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.info("fetch failed %s: %s", url, e)
        return None
    return resp.content if as_bytes else resp.text


def _has_class(cls: str) -> str:
    return f"contains(concat(' ', normalize-space(@class), ' '), ' {cls} ')"


def _parse_arxiv_html(html: str) -> str | None:
    "Structure-aware extraction from arXiv (LatexML) HTML -> markdown."
    try:
        tree = lxml.html.fromstring(html)
    except Exception:
        return None

    root = tree.xpath(f".//*[{_has_class('ltx_page_content')}]")
    root = root[0] if root else tree

    # drop non-content section (bibliography, appendix, ...)
    drop_expr = " or ".join(_has_class(c) for c in _LTX_DROP)
    for node in root.xpath(f".//*[{drop_expr}]"):
        node.getparent().remove(node)

    # headings + paragraphs -> markdown
    want = list(_LTX_LEVEL) + ["ltx_p"]
    want_expr = " or ".join(_has_class(c) for c in want)

    parts: list[str] = []
    for node in root.xpath(f".//*[{want_expr}]"):
        classes = (node.get("class") or "").split()
        level = next((_LTX_LEVEL[c] for c in classes if c in _LTX_LEVEL), None)
        text = " ".join(node.text_content().split())
        if not text:
            continue
        parts.append(f"{level} {text}" if level else text)

    return "\n\n".join(parts) or None

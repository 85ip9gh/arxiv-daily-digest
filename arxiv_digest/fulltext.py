"""Pulls the body of a paper, not just its abstract.

arXiv renders most submissions since late 2023 to HTML at `arxiv.org/html/<id>`.
That is the cheap path to a paper's actual method and results, and it is why the
summaries can carry architecture, datasets and measured numbers instead of the
abstract's marketing sentence.

Older or PDF-only submissions have no HTML rendering. Those fall back to the
abstract, and the digest says so on the page rather than implying a depth of
reading that did not happen.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

import requests

from .arxiv import USER_AGENT, Paper

HTML_URL = "https://arxiv.org/html/{arxiv_id}{version}"
DEFAULT_TIMEOUT = 45
# Roughly 3.5k tokens of paper. Free model tiers meter tokens per minute, and
# three summaries have to fit inside one minute's allowance with room for the
# prompt and the answer.
DEFAULT_BUDGET = 14000

# Skipped whole: `math` is LaTeXML markup whose text content is unreadable noise,
# and the reference list is a wall of names that crowds out the method.
SKIP_TAGS = {"script", "style", "math", "svg", "noscript"}
SKIP_HEADINGS = re.compile(
    r"^\s*(references|bibliography|acknowledg|appendix|author contribution|"
    r"ethics statement|impact statement|checklist)",
    re.I,
)

# Section titles worth spending the budget on, most valuable first. A paper's
# introduction restates the abstract; its method and results do not.
PRIORITY = (
    (re.compile(r"method|approach|architecture|model|framework|design|algorithm", re.I), 5),
    (re.compile(r"experiment|evaluation|result|benchmark|ablation|analysis", re.I), 4),
    (re.compile(r"implementation|training|dataset|data|setup", re.I), 3),
    (re.compile(r"limitation|discussion|conclusion", re.I), 2),
    (re.compile(r"introduction|background|related work", re.I), 1),
)

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class Section:
    heading: str
    text: str

    @property
    def score(self) -> int:
        for pattern, weight in PRIORITY:
            if pattern.search(self.heading):
                return weight
        return 1


class _Reader(HTMLParser):
    """Collects text per section heading. Deliberately forgiving: a parse that
    returns less text is better than one that raises on a malformed tag."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[Section] = []
        self._heading = "Body"
        self._chunks: list[str] = []
        self._skip_depth = 0
        self._in_heading = False
        self._heading_parts: list[str] = []

    def _flush(self) -> None:
        text = _BLANKS.sub("\n\n", "".join(self._chunks)).strip()
        if text:
            self.sections.append(Section(self._heading, text))
        self._chunks = []

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in {"h1", "h2", "h3"}:
            self._flush()
            self._in_heading = True
            self._heading_parts = []
        elif tag in {"p", "li", "tr", "div", "br", "td", "th"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag in {"h1", "h2", "h3"} and self._in_heading:
            self._in_heading = False
            self._heading = _WHITESPACE.sub(" ", "".join(self._heading_parts)).strip() or "Body"
        elif tag in {"td", "th"}:
            self._chunks.append(" ")

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_heading:
            self._heading_parts.append(data)
        else:
            self._chunks.append(_WHITESPACE.sub(" ", data))

    def close(self):
        super().close()
        self._flush()


def parse_sections(html_text: str) -> list[Section]:
    reader = _Reader()
    try:
        reader.feed(html_text)
        reader.close()
    except Exception:
        # A half-parsed paper still summarizes; a raised exception at 07:00
        # loses the whole day.
        pass
    return [s for s in reader.sections if not SKIP_HEADINGS.match(s.heading)]


def pack(sections: list[Section], budget: int = DEFAULT_BUDGET) -> str:
    """Fit the most useful sections into the budget, in document order.

    Selection is by section value, assembly is by document order, so the model
    reads method before results even when results scored higher.

    No single section may take more than a third of the budget. A long method
    section that swallowed the whole allowance would leave the summary with a
    detailed mechanism and no measured outcome, which is the wrong half to keep.
    """
    if not sections:
        return ""

    cap = max(1200, budget // 3)
    ordered = sorted(range(len(sections)), key=lambda i: (-sections[i].score, i))
    kept: dict[int, str] = {}
    used = 0

    for i in ordered:
        room = min(cap, budget - used)
        if room < 400:
            break
        chunk = sections[i]
        text = chunk.text
        if len(text) > room:
            text = text[:room].rsplit(" ", 1)[0] + " [...]"
        kept[i] = text
        used += len(text) + len(chunk.heading) + 4

    parts = [
        f"## {sections[i].heading}\n{kept[i]}" for i in sorted(kept) if kept[i].strip()
    ]
    return "\n\n".join(parts)[:budget]


def fetch(
    paper: Paper, *, budget: int = DEFAULT_BUDGET, timeout: int = DEFAULT_TIMEOUT
) -> str | None:
    """Return the packed body text, or None when arXiv has no HTML rendering."""
    url = HTML_URL.format(arxiv_id=paper.arxiv_id, version=paper.version)
    try:
        response = requests.get(
            url, timeout=timeout, headers={"User-Agent": USER_AGENT}
        )
    except requests.RequestException:
        return None
    if response.status_code != 200 or "html" not in response.headers.get(
        "Content-Type", ""
    ):
        return None

    packed = pack(parse_sections(response.text), budget)
    # A rendering that yields almost nothing is a redirect page or a stub, and
    # pretending it is a paper produces a confidently empty summary.
    return packed if len(packed) > 2000 else None

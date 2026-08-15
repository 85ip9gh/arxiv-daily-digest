"""Static site rendered from the archive: one index, one page per day.

Deliberately dependency free and asset free. Every page is one self-contained
HTML file with its CSS inline, so the whole site is a directory nginx can serve
read-only, with nothing to build and nothing to fetch at page load.
"""
from __future__ import annotations

import html
from datetime import date
from pathlib import Path

SITE_TITLE = "arXiv AI digest"
SITE_TAGLINE = "Three new AI papers a day, summarized and checked against their abstracts."

STYLE = """
:root {
  color-scheme: light dark;
  --bg: #fcfcfa;
  --panel: #ffffff;
  --ink: #17181a;
  --muted: #6b6f76;
  --line: #e3e3df;
  --accent: #17181a;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111214;
    --panel: #17181a;
    --ink: #ededea;
    --muted: #9aa0a6;
    --line: #2a2c30;
    --accent: #ededea;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 3rem 1.25rem 5rem;
  background: var(--bg);
  color: var(--ink);
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
main { max-width: 46rem; margin: 0 auto; }
a { color: var(--accent); text-underline-offset: 2px; }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
header.masthead { border-bottom: 1px solid var(--line); padding-bottom: 1.25rem; margin-bottom: 2rem; }
header.masthead h1 { font-size: 1.5rem; margin: 0 0 .35rem; letter-spacing: -.01em; }
header.masthead p { margin: 0; color: var(--muted); font-size: .95rem; }
.back { display: inline-block; margin-bottom: 1.5rem; font-size: .9rem; color: var(--muted); }
.day-list { list-style: none; margin: 0; padding: 0; }
.day-list li { border-bottom: 1px solid var(--line); padding: 1.25rem 0; }
.day-list h2 { font-size: 1.05rem; margin: 0 0 .5rem; }
.day-list ol { margin: 0; padding-left: 1.1rem; color: var(--muted); font-size: .92rem; }
.day-list ol li { border: 0; padding: .15rem 0; }
article { border-top: 1px solid var(--line); padding-top: 1.75rem; margin-top: 1.75rem; }
article:first-of-type { border-top: 0; margin-top: 0; }
article h2 { font-size: 1.15rem; line-height: 1.35; margin: 0 0 .4rem; }
.meta { color: var(--muted); font-size: .85rem; margin: 0 0 1rem; }
.meta a { color: var(--muted); }
.field { margin: 0 0 .85rem; }
.field b { font-weight: 600; }
blockquote {
  margin: 1.25rem 0 .75rem;
  padding: .6rem 0 .6rem 1rem;
  border-left: 3px solid var(--line);
  color: var(--muted);
  font-size: .93rem;
}
blockquote.unverified { border-left-color: #b4632a; }
.reason { color: var(--muted); font-size: .85rem; font-style: italic; margin: 0; }
nav.pager { display: flex; justify-content: space-between; gap: 1rem; margin-top: 3rem; padding-top: 1.25rem; border-top: 1px solid var(--line); font-size: .9rem; }
footer { margin-top: 3rem; padding-top: 1.25rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .82rem; }
"""


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _long_date(iso: str) -> str:
    """`2026-08-15` to `August 15, 2026`. The day is built by hand because the
    no-pad format code differs between glibc and Windows."""
    try:
        parsed = date.fromisoformat(iso)
    except (ValueError, TypeError):
        return str(iso)
    return f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}"


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_e(title)}</title>\n"
        f'<meta name="description" content="{_e(SITE_TAGLINE)}">\n'
        f"<style>{STYLE}</style>\n"
        "</head>\n<body>\n<main>\n"
        f"{body}"
        "</main>\n</body>\n</html>\n"
    )


def _masthead(subtitle: str) -> str:
    return (
        '<header class="masthead">\n'
        f"<h1>{_e(SITE_TITLE)}</h1>\n"
        f"<p>{_e(subtitle)}</p>\n"
        "</header>\n"
    )


def _footer(day: dict | None = None) -> str:
    bits = ["Summaries are machine written from the arXiv abstract and quote-checked against it."]
    if day:
        bits.append(f"Model: {day.get('model', 'unknown')}.")
        if day.get("generated_at"):
            bits.append(f"Generated {day['generated_at']}.")
    return f"<footer>{_e(' '.join(bits))}</footer>\n"


def render_index(days: list[dict]) -> str:
    if not days:
        body = _masthead(SITE_TAGLINE) + "<p>No digests yet.</p>\n" + _footer()
        return _page(SITE_TITLE, body)

    items = []
    for day in days:
        titles = "".join(
            f"<li>{_e(p.get('title', ''))}</li>\n" for p in day.get("papers", [])
        )
        items.append(
            "<li>\n"
            f'<h2><a href="{_e(day["date"])}.html">{_e(_long_date(day["date"]))}</a></h2>\n'
            f"<ol>\n{titles}</ol>\n"
            "</li>\n"
        )

    body = (
        _masthead(SITE_TAGLINE)
        + f'<ul class="day-list">\n{"".join(items)}</ul>\n'
        + _footer()
    )
    return _page(SITE_TITLE, body)


def render_day(day: dict, *, newer: str | None = None, older: str | None = None) -> str:
    articles = []
    for paper in day.get("papers", []):
        quote = paper.get("quote", "")
        if paper.get("grounded") and quote:
            quote_html = f"<blockquote>{_e(quote)}</blockquote>\n"
        else:
            quote_html = (
                '<blockquote class="unverified">Citation check failed. The model '
                "could not quote the abstract for its claim, so treat this summary "
                "as unverified.</blockquote>\n"
            )
        reason = paper.get("reason", "")
        reason_html = f'<p class="reason">Picked because: {_e(reason)}</p>\n' if reason else ""
        articles.append(
            "<article>\n"
            f'<h2><a href="{_e(paper.get("abs_url", "#"))}">{_e(paper.get("title", ""))}</a></h2>\n'
            f'<p class="meta">{_e(paper.get("author_line", ""))} &middot; '
            f'{_e(paper.get("primary_category", ""))} &middot; '
            f'<a href="{_e(paper.get("abs_url", "#"))}">abstract</a> &middot; '
            f'<a href="{_e(paper.get("pdf_url", "#"))}">pdf</a></p>\n'
            f'<p class="field"><b>Problem.</b> {_e(paper.get("problem", ""))}</p>\n'
            f'<p class="field"><b>Approach.</b> {_e(paper.get("approach", ""))}</p>\n'
            f'<p class="field"><b>Result.</b> {_e(paper.get("result", ""))}</p>\n'
            f'<p class="field"><b>Why it matters.</b> {_e(paper.get("so_what", ""))}</p>\n'
            f"{quote_html}{reason_html}"
            "</article>\n"
        )

    pager = ""
    if newer or older:
        left = f'<a href="{_e(older)}.html">Older: {_e(_long_date(older))}</a>' if older else "<span></span>"
        right = f'<a href="{_e(newer)}.html">Newer: {_e(_long_date(newer))}</a>' if newer else "<span></span>"
        pager = f'<nav class="pager">{left}{right}</nav>\n'

    body = (
        '<a class="back" href="index.html">All days</a>\n'
        + _masthead(_long_date(day["date"]))
        + "".join(articles)
        + pager
        + _footer(day)
    )
    return _page(f"{_long_date(day['date'])} | {SITE_TITLE}", body)


def build(days: list[dict], site_dir: Path) -> list[Path]:
    """Write the whole site. Every page is regenerated on every build."""
    site_dir.mkdir(parents=True, exist_ok=True)
    written = [site_dir / "index.html"]
    (site_dir / "index.html").write_text(render_index(days), encoding="utf-8")

    # `days` is newest first, so the previous entry is the newer neighbour.
    for i, day in enumerate(days):
        newer = days[i - 1]["date"] if i > 0 else None
        older = days[i + 1]["date"] if i + 1 < len(days) else None
        path = site_dir / f"{day['date']}.html"
        path.write_text(render_day(day, newer=newer, older=older), encoding="utf-8")
        written.append(path)
    return written

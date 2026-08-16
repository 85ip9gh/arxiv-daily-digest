"""Static site rendered from the archive: one index, one page per day.

Dependency free and asset free. Every page is one self-contained HTML file with
its CSS and its script inline, so the whole site is a directory nginx serves
read-only, with nothing to build and nothing fetched at page load.

The design treats a summary as a lab record rather than a blog post. What a
reader needs first is not the prose, it is whether the prose was checked, so the
evidence chips sit in the meta row next to the authors and the quote is set as a
citation rather than a pull quote. Detail that only some readers want (method
specifics, the figures, the limitations) is behind native disclosures, which
keeps the morning skim short without hiding anything.
"""
from __future__ import annotations

import html
from datetime import date
from pathlib import Path

SITE_TITLE = "arXiv AI digest"
TAGLINE_TAIL = (
    "read in full where arXiv renders them, and checked against the source."
)
# How far back the tagline looks when counting. The number is read from the
# archive rather than written down, because a hardcoded "three" outlived the
# three paper era by exactly one config change and told every visitor the wrong
# thing. One short day cannot drag it down, and a deliberate change to the daily
# count shows up within a week.
TAGLINE_WINDOW = 7

_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}


def tagline(days: list[dict]) -> str:
    """Describe the daily cadence using what the archive actually contains.

    "Up to" rather than a flat count, because a run that loses a paper to the
    token budget publishes a short day on purpose.
    """
    counts = [len(d.get("papers", [])) for d in days[:TAGLINE_WINDOW]]
    counts = [c for c in counts if c > 0]
    if not counts:
        return f"New AI papers every morning, {TAGLINE_TAIL}"
    highest = max(counts)
    if highest == 1:
        return f"One new AI paper a day, {TAGLINE_TAIL}"
    word = _NUMBER_WORDS.get(highest, str(highest))
    return f"Up to {word} new AI papers a day, {TAGLINE_TAIL}"

# Machine-written summaries sitting under a personal domain would compete with
# that domain's own pages in search results. The site stays publicly readable,
# it just does not ask to be indexed. Delete this to opt back in.
ROBOTS = "User-agent: *\nDisallow: /\n"

STYLE = """
:root {
  --ground: #f7f7f5;
  --panel: #ffffff;
  --ink: #171717;
  --muted: #525252;
  --faint: #6f6f6f;
  --line: #d8d8d4;
  --line-strong: #c6c6c1;
  --accent: #292929;
  --accent-soft: #ececea;
  --ok: #1f7a4d;
  --ok-soft: #e6f1eb;
  --warn: #8a6410;
  --warn-soft: #f5ece0;
  --shadow: 0 0.4rem 1.2rem rgba(17, 17, 19, .04);
  --display: "Geist", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --body: "Geist", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "Geist Mono", ui-monospace, SFMono-Regular, "Cascadia Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #101111;
    --panel: #181919;
    --ink: #f1f1ee;
    --muted: #bebebb;
    --faint: #999995;
    --line: #373838;
    --line-strong: #4a4b4b;
    --accent: #e6e6e2;
    --accent-soft: #272828;
    --ok: #4fbf85;
    --ok-soft: #16261e;
    --warn: #d4a029;
    --warn-soft: #2a2113;
    --shadow: 0 0.4rem 1.2rem rgba(0, 0, 0, .35);
  }
}
:root[data-theme="dark"] {
  --ground: #101111;
  --panel: #181919;
  --ink: #f1f1ee;
  --muted: #bebebb;
  --faint: #999995;
  --line: #373838;
  --line-strong: #4a4b4b;
  --accent: #e6e6e2;
  --accent-soft: #272828;
  --ok: #4fbf85;
  --ok-soft: #16261e;
  --warn: #d4a029;
  --warn-soft: #2a2113;
  --shadow: 0 0.4rem 1.2rem rgba(0, 0, 0, .35);
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: var(--body);
  font-size: 16px;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }
a:hover { text-decoration-thickness: 2px; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 2px; }

.wrap { max-width: 54rem; margin: 0 auto; padding: 0 1.25rem 5rem; }

/* Masthead ---------------------------------------------------------------- */
.top {
  position: sticky; top: 0; z-index: 10;
  background: color-mix(in srgb, var(--ground) 88%, transparent);
  backdrop-filter: saturate(160%) blur(8px);
  border-bottom: 1px solid var(--line);
}
.top-inner {
  max-width: 54rem; margin: 0 auto; padding: .6rem 1.25rem;
  display: flex; align-items: center; gap: 1rem;
}
.brand {
  font-family: var(--mono); font-size: .78rem; letter-spacing: .08em;
  text-transform: uppercase; color: var(--muted); text-decoration: none;
  white-space: nowrap;
}
.brand b { color: var(--ink); font-weight: 600; }
.top nav { margin-left: auto; display: flex; align-items: center; gap: .35rem; flex-wrap: wrap; }
.jump {
  font-family: var(--mono); font-size: .75rem; color: var(--muted);
  text-decoration: none; padding: .25rem .45rem; border-radius: 0.35rem;
}
.jump:hover { background: var(--accent-soft); color: var(--accent); }

button.ctl {
  font-family: var(--mono); font-size: .72rem; letter-spacing: .04em;
  color: var(--muted); background: transparent;
  border: 1px solid var(--line-strong); border-radius: 0.35rem;
  padding: .25rem .5rem; cursor: pointer;
}
button.ctl:hover { color: var(--ink); border-color: var(--ink); }

header.masthead { padding: 3rem 0 1.5rem; }
header.masthead h1 {
  font-family: var(--display); font-weight: 650; font-size: clamp(1.9rem, 5vw, 2.6rem);
  line-height: 1.06; margin: 0 0 .6rem; letter-spacing: -.045em; text-wrap: balance;
}
header.masthead p { margin: 0; color: var(--muted); max-width: 46ch; }
header.masthead p.eyebrow,
.eyebrow {
  font-family: var(--mono); font-size: .68rem; font-weight: 600; letter-spacing: .12em;
  text-transform: uppercase; color: var(--accent); margin: 0 0 .75rem;
}

/* Filter ------------------------------------------------------------------ */
.filter { display: flex; align-items: baseline; gap: .75rem; margin: 2rem 0 .5rem; flex-wrap: wrap; }
.filter input {
  flex: 1 1 16rem; min-width: 0;
  font-family: var(--body); font-size: .95rem; color: var(--ink);
  background: var(--panel); border: 1px solid var(--line-strong);
  border-radius: 0.35rem; padding: .5rem .7rem;
}
.filter input::placeholder { color: var(--faint); }
.count { font-family: var(--mono); font-size: .76rem; color: var(--muted); white-space: nowrap; }

/* Day ledger -------------------------------------------------------------- */
.ledger { list-style: none; margin: 0; padding: 0; border-top: 1px solid var(--line); }
.ledger li { border-bottom: 1px solid var(--line); }
.ledger li[hidden] { display: none; }
.entry {
  display: grid; grid-template-columns: 8.5rem 1fr; gap: 1.5rem;
  padding: 1.4rem .5rem; text-decoration: none; color: inherit;
  transition: background 140ms ease;
}
.entry:hover { background: var(--panel); }
.entry-date { font-family: var(--mono); font-size: .82rem; color: var(--accent); line-height: 1.5; }
.entry-date span { display: block; color: var(--faint); font-size: .74rem; }
.entry ol { margin: 0; padding: 0; list-style: none; counter-reset: t; }
.entry ol li {
  border: 0; padding: .1rem 0 .1rem 1.6rem; position: relative;
  font-family: var(--display); font-size: 1.02rem; line-height: 1.45; text-wrap: pretty;
}
.entry ol li::before {
  counter-increment: t; content: counter(t);
  position: absolute; left: 0; top: .28rem;
  font-family: var(--mono); font-size: .7rem; color: var(--faint);
}
.tags { margin-top: .55rem; display: flex; gap: .3rem; flex-wrap: wrap; }

/* Chips ------------------------------------------------------------------- */
.chip {
  font-family: var(--mono); font-size: .62rem; letter-spacing: .04em;
  padding: .15rem .45rem; border-radius: 999px; white-space: nowrap;
  border: 1px solid var(--line-strong); color: var(--muted);
}
.chip.ok { color: var(--ok); background: var(--ok-soft); border-color: transparent; }
.chip.warn { color: var(--warn); background: var(--warn-soft); border-color: transparent; }
.chip.cat { color: var(--accent); background: var(--accent-soft); border-color: transparent; }

/* Paper record ------------------------------------------------------------ */
article {
  background: var(--panel); border: 1px solid var(--line); border-radius: 0.6rem;
  box-shadow: var(--shadow); padding: 1.75rem; margin: 0 0 1.5rem;
  scroll-margin-top: 4.5rem;
}
.rank {
  font-family: var(--mono); font-size: .72rem; letter-spacing: .1em;
  color: var(--faint); text-transform: uppercase; margin: 0 0 .5rem;
}
article h2 {
  font-family: var(--display); font-weight: 650; font-size: 1.42rem; line-height: 1.22;
  margin: 0 0 .6rem; letter-spacing: -.03em; text-wrap: balance;
}
article h2 a { color: inherit; text-decoration: none; }
article h2 a:hover { color: var(--accent); }
.meta { font-size: .86rem; color: var(--muted); margin: 0 0 .9rem; }
.meta .authors { font-style: italic; }
.badges { display: flex; gap: .35rem; flex-wrap: wrap; margin: 0 0 1.25rem; align-items: center; }
.field { margin: 0 0 1rem; }
.field h3 {
  font-family: var(--mono); font-size: .72rem; letter-spacing: .1em;
  text-transform: uppercase; color: var(--faint); margin: 0 0 .25rem; font-weight: 500;
}
.field p { margin: 0; text-wrap: pretty; }

details {
  border-top: 1px solid var(--line); padding: .7rem 0 0; margin: 0 0 .7rem;
}
details > summary {
  font-family: var(--mono); font-size: .74rem; letter-spacing: .06em;
  text-transform: uppercase; color: var(--muted); cursor: pointer; list-style: none;
  display: flex; align-items: center; gap: .4rem;
}
details > summary::-webkit-details-marker { display: none; }
details > summary::before {
  content: "+"; font-size: .9rem; color: var(--faint); width: .7rem;
}
details[open] > summary::before { content: "\\2212"; }
details > summary:hover { color: var(--ink); }
details .body { padding: .6rem 0 .2rem 1.1rem; }
details ul { margin: 0; padding-left: 1.1rem; }
details li { margin: .3rem 0; text-wrap: pretty; }

.figures { margin: 0; padding: 0; list-style: none; }
.figures li {
  font-family: var(--mono); font-size: .84rem; font-variant-numeric: tabular-nums;
  padding: .3rem 0; border-bottom: 1px dotted var(--line); text-wrap: pretty;
}
.figures li:last-child { border-bottom: 0; }

blockquote {
  margin: 1.25rem 0 .5rem; padding: .5rem 0 .5rem 1rem;
  border-left: 2px solid var(--ok); color: var(--muted);
  font-size: .92rem; font-style: italic; text-wrap: pretty;
}
blockquote.unverified { border-left-color: var(--warn); font-style: normal; }
blockquote cite { display: block; font-style: normal; font-family: var(--mono); font-size: .7rem; color: var(--faint); margin-top: .4rem; }
.reason { color: var(--faint); font-size: .82rem; margin: .9rem 0 0; text-wrap: pretty; }

/* Pager and footer -------------------------------------------------------- */
nav.pager { display: flex; justify-content: space-between; gap: 1rem; margin-top: 2.5rem; font-size: .9rem; }
nav.pager a { font-family: var(--mono); font-size: .8rem; text-decoration: none; }
nav.pager a:hover { text-decoration: underline; }
footer {
  margin-top: 3.5rem; padding-top: 1.25rem; border-top: 1px solid var(--line);
  color: var(--faint); font-size: .78rem; font-family: var(--mono); line-height: 1.8;
}
footer kbd {
  font-family: var(--mono); font-size: .72rem; border: 1px solid var(--line-strong);
  border-bottom-width: 2px; border-radius: 3px; padding: 0 .25rem; color: var(--muted);
}
.empty { color: var(--muted); padding: 2rem 0; }

@media (max-width: 34rem) {
  .entry { grid-template-columns: 1fr; gap: .5rem; padding: 1.1rem .25rem; }
  article { padding: 1.25rem; }
}
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; scroll-behavior: auto !important; }
}
"""

# Runs before first paint so an explicit theme choice never flashes the other one.
THEME_BOOT = (
    "(function(){try{var t=localStorage.getItem('digest-theme');"
    "if(t==='dark'||t==='light'){document.documentElement.setAttribute('data-theme',t);}}"
    "catch(e){}})();"
)

THEME_SCRIPT = """
(function () {
  var root = document.documentElement;
  var btn = document.getElementById('theme');
  if (!btn) return;
  function label() {
    var t = root.getAttribute('data-theme');
    btn.textContent = t === 'dark' ? 'dark' : t === 'light' ? 'light' : 'system';
    btn.setAttribute('aria-label', 'Theme: ' + btn.textContent + '. Click to change.');
  }
  btn.addEventListener('click', function () {
    var order = ['system', 'light', 'dark'];
    var now = root.getAttribute('data-theme') || 'system';
    var next = order[(order.indexOf(now) + 1) % order.length];
    if (next === 'system') { root.removeAttribute('data-theme'); }
    else { root.setAttribute('data-theme', next); }
    try { localStorage.setItem('digest-theme', next); } catch (e) {}
    label();
  });
  label();
})();
"""

INDEX_SCRIPT = """
(function () {
  var box = document.getElementById('filter');
  var count = document.getElementById('count');
  // Direct children only. The nested list of paper titles is also made of
  // list items, and counting those reports four days for one.
  var rows = Array.prototype.slice.call(document.querySelectorAll('.ledger > li'));
  if (!box) return;
  function apply() {
    var q = box.value.trim().toLowerCase();
    var shown = 0;
    rows.forEach(function (row) {
      var hit = !q || (row.dataset.text || '').indexOf(q) !== -1;
      row.hidden = !hit;
      if (hit) shown++;
    });
    count.textContent = shown === rows.length
      ? rows.length + (rows.length === 1 ? ' day' : ' days')
      : shown + ' of ' + rows.length;
  }
  box.addEventListener('input', apply);
  document.addEventListener('keydown', function (e) {
    if (e.key === '/' && document.activeElement !== box) { e.preventDefault(); box.focus(); }
    if (e.key === 'Escape' && document.activeElement === box) { box.value = ''; apply(); box.blur(); }
  });
  apply();
})();
"""

DAY_SCRIPT = """
(function () {
  var papers = Array.prototype.slice.call(document.querySelectorAll('article'));
  var expand = document.getElementById('expand');
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (expand) {
    expand.addEventListener('click', function () {
      var open = expand.dataset.state !== 'open';
      document.querySelectorAll('details').forEach(function (d) { d.open = open; });
      expand.dataset.state = open ? 'open' : 'closed';
      expand.textContent = open ? 'collapse all' : 'expand all';
    });
  }

  document.querySelectorAll('button.copy').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var text = btn.dataset.cite;
      var done = function () {
        var was = btn.textContent;
        btn.textContent = 'copied';
        setTimeout(function () { btn.textContent = was; }, 1400);
      };
      if (navigator.clipboard) { navigator.clipboard.writeText(text).then(done, done); }
      else { done(); }
    });
  });

  function go(delta) {
    var top = window.scrollY + 80;
    var i = 0;
    papers.forEach(function (p, n) { if (p.offsetTop <= top) i = n; });
    var target = papers[Math.min(papers.length - 1, Math.max(0, i + delta))];
    if (target) target.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
  }
  document.addEventListener('keydown', function (e) {
    var tag = (document.activeElement || {}).tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'j') { e.preventDefault(); go(1); }
    if (e.key === 'k') { e.preventDefault(); go(-1); }
    var link = e.key === ',' ? document.getElementById('older')
             : e.key === '.' ? document.getElementById('newer') : null;
    if (link) { e.preventDefault(); window.location.href = link.href; }
  });
})();
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


def _weekday(iso: str) -> str:
    try:
        return date.fromisoformat(iso).strftime("%A")
    except (ValueError, TypeError):
        return ""


def _page(title: str, body: str, scripts: str, description: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="robots" content="noindex">\n'
        '<meta name="color-scheme" content="light dark">\n'
        f"<title>{_e(title)}</title>\n"
        f'<meta name="description" content="{_e(description)}">\n'
        f"<style>{STYLE}</style>\n"
        f"<script>{THEME_BOOT}</script>\n"
        "</head>\n<body>\n"
        f"{body}"
        f"<script>{THEME_SCRIPT}{scripts}</script>\n"
        "</body>\n</html>\n"
    )


def _topbar(links: str = "") -> str:
    return (
        '<div class="top"><div class="top-inner">\n'
        f'<a class="brand" href="index.html">arXiv <b>digest</b></a>\n'
        f"<nav>{links}"
        '<button class="ctl" id="theme" type="button">system</button>'
        "</nav>\n</div></div>\n"
    )


def _footer(day: dict | None = None, keys: str = "") -> str:
    lines = [
        "Summaries are machine written from the paper text and checked against it."
    ]
    if day:
        lines.append(f"Model: {_e(day.get('model', 'unknown'))}.")
        if day.get("generated_at"):
            lines.append(f"Generated {_e(day['generated_at'])}.")
    return f"<footer>{' '.join(lines)}{keys}</footer>\n"


def render_index(days: list[dict]) -> str:
    blurb = tagline(days)
    head = (
        _topbar()
        + '<div class="wrap">\n<header class="masthead">\n'
        '<p class="eyebrow">Daily, 07:00 Atlantic</p>\n'
        f"<h1>{_e(SITE_TITLE)}</h1>\n"
        f"<p>{_e(blurb)}</p>\n</header>\n"
    )

    if not days:
        body = head + '<p class="empty">No digests yet.</p>\n' + _footer() + "</div>\n"
        return _page(SITE_TITLE, body, "", blurb)

    rows = []
    for day in days:
        papers = day.get("papers", [])
        titles = "".join(f"<li>{_e(p.get('title', ''))}</li>\n" for p in papers)
        cats = []
        for cat in dict.fromkeys(p.get("primary_category", "") for p in papers):
            if cat:
                cats.append(f'<span class="chip cat">{_e(cat)}</span>')
        haystack = " ".join(
            [day["date"], _long_date(day["date"]), _weekday(day["date"])]
            + [str(p.get("title", "")) for p in papers]
            + [str(p.get("primary_category", "")) for p in papers]
        ).lower()
        rows.append(
            f'<li data-text="{_e(haystack)}">\n'
            f'<a class="entry" href="{_e(day["date"])}.html">\n'
            f'<div class="entry-date">{_e(_long_date(day["date"]))}'
            f"<span>{_e(_weekday(day['date']))}</span></div>\n"
            f"<div><ol>{titles}</ol>\n"
            f'<div class="tags">{"".join(cats)}</div></div>\n'
            "</a>\n</li>\n"
        )

    body = (
        head
        + '<div class="filter">\n'
        '<input id="filter" type="search" placeholder="Filter by title, category or date" '
        'autocomplete="off" aria-label="Filter digests">\n'
        f'<span class="count" id="count" role="status">{len(days)} '
        f'{"day" if len(days) == 1 else "days"}</span>\n'
        "</div>\n"
        f'<ul class="ledger">\n{"".join(rows)}</ul>\n'
        + _footer(keys=" Press <kbd>/</kbd> to filter.")
        + "</div>\n"
    )
    return _page(SITE_TITLE, body, INDEX_SCRIPT, blurb)


def _badges(paper: dict) -> str:
    chips = []
    if paper.get("grounded") and paper.get("quote"):
        chips.append('<span class="chip ok">quote verified</span>')
    else:
        chips.append('<span class="chip warn">quote unverified</span>')

    stray = paper.get("unverified_numbers") or []
    if stray:
        chips.append(
            f'<span class="chip warn">{len(stray)} figure'
            f'{"s" if len(stray) > 1 else ""} not in source</span>'
        )
    else:
        chips.append('<span class="chip ok">figures checked</span>')

    label = paper.get("source_label") or (
        "full text" if paper.get("read_full_text") else "abstract only"
    )
    chips.append(f'<span class="chip">read: {_e(label)}</span>')
    if paper.get("primary_category"):
        chips.append(f'<span class="chip cat">{_e(paper["primary_category"])}</span>')
    return "".join(chips)


def _details(title: str, inner: str, *, open_by_default: bool = False) -> str:
    if not inner:
        return ""
    attr = " open" if open_by_default else ""
    return (
        f"<details{attr}><summary>{_e(title)}</summary>\n"
        f'<div class="body">{inner}</div>\n</details>\n'
    )


def _article(index: int, paper: dict) -> str:
    method = "".join(f"<li>{_e(d)}</li>\n" for d in paper.get("method_details") or [])
    figures = "".join(f"<li>{_e(n)}</li>\n" for n in paper.get("numbers") or [])
    stray = paper.get("unverified_numbers") or []

    quote = paper.get("quote", "")
    if paper.get("grounded") and quote:
        quote_html = (
            f"<blockquote>{_e(quote)}"
            "<cite>Found in the source text, word for word.</cite></blockquote>\n"
        )
    else:
        quote_html = (
            '<blockquote class="unverified">The model could not quote the source '
            "for its claim, so nothing above has been checked against the paper. "
            "Read the abstract before trusting it."
            "<cite>Citation check failed.</cite></blockquote>\n"
        )

    stray_html = ""
    if stray:
        stray_html = (
            '<blockquote class="unverified">These figures do not appear in the '
            f"source text: {_e(', '.join(stray))}. Treat them as unverified."
            "<cite>Number check failed.</cite></blockquote>\n"
        )

    cite_text = f"{paper.get('title', '')} ({paper.get('abs_url', '')})"
    reason = paper.get("reason", "")
    reason_html = (
        f'<p class="reason">Picked because: {_e(reason)}</p>\n' if reason else ""
    )

    return (
        f'<article id="p{index}">\n'
        f'<p class="rank">Paper {index} of 3</p>\n'
        f'<h2><a href="{_e(paper.get("abs_url", "#"))}">{_e(paper.get("title", ""))}</a></h2>\n'
        f'<p class="meta"><span class="authors">{_e(paper.get("author_line", ""))}</span> '
        f'&middot; <a href="{_e(paper.get("abs_url", "#"))}">abstract</a> '
        f'&middot; <a href="{_e(paper.get("pdf_url", "#"))}">pdf</a></p>\n'
        f'<div class="badges">{_badges(paper)}'
        f'<button class="ctl copy" type="button" data-cite="{_e(cite_text)}">copy link</button>'
        "</div>\n"
        f'<div class="field"><h3>Problem</h3><p>{_e(paper.get("problem", ""))}</p></div>\n'
        f'<div class="field"><h3>Approach</h3><p>{_e(paper.get("approach", ""))}</p></div>\n'
        f'<div class="field"><h3>Result</h3><p>{_e(paper.get("result", ""))}</p></div>\n'
        f'<div class="field"><h3>Why it matters</h3><p>{_e(paper.get("so_what", ""))}</p></div>\n'
        + _details("Method details", f"<ul>{method}</ul>" if method else "")
        + _details(
            "Numbers", f'<ul class="figures">{figures}</ul>' if figures else ""
        )
        + _details(
            "Limitations",
            f"<p>{_e(paper['limitations'])}</p>" if paper.get("limitations") else "",
        )
        + quote_html
        + stray_html
        + reason_html
        + "</article>\n"
    )


def render_day(
    day: dict,
    *,
    newer: str | None = None,
    older: str | None = None,
    blurb: str | None = None,
) -> str:
    # The blurb comes from the whole archive so every page says the same thing.
    # Falling back to this one day keeps render_day usable on its own.
    blurb = blurb if blurb is not None else tagline([day])
    papers = day.get("papers", [])
    jumps = "".join(
        f'<a class="jump" href="#p{i}">{i}</a>' for i in range(1, len(papers) + 1)
    )
    jumps += '<button class="ctl" id="expand" type="button" data-state="closed">expand all</button>'

    articles = "".join(_article(i, p) for i, p in enumerate(papers, start=1))

    pager = ""
    if newer or older:
        left = (
            f'<a id="older" href="{_e(older)}.html">&larr; {_e(_long_date(older))}</a>'
            if older
            else "<span></span>"
        )
        right = (
            f'<a id="newer" href="{_e(newer)}.html">{_e(_long_date(newer))} &rarr;</a>'
            if newer
            else "<span></span>"
        )
        pager = f'<nav class="pager">{left}{right}</nav>\n'

    body = (
        _topbar(jumps)
        + '<div class="wrap">\n<header class="masthead">\n'
        f'<p class="eyebrow">{_e(_weekday(day["date"]))}</p>\n'
        f"<h1>{_e(_long_date(day['date']))}</h1>\n"
        f"<p>{_e(blurb)}</p>\n</header>\n"
        + articles
        + pager
        + _footer(
            day,
            " Keys: <kbd>j</kbd> <kbd>k</kbd> between papers, "
            "<kbd>,</kbd> <kbd>.</kbd> between days.",
        )
        + "</div>\n"
    )
    return _page(f"{_long_date(day['date'])} | {SITE_TITLE}", body, DAY_SCRIPT, blurb)


def build(days: list[dict], site_dir: Path) -> list[Path]:
    """Write the whole site. Every page is regenerated on every build."""
    site_dir.mkdir(parents=True, exist_ok=True)
    blurb = tagline(days)
    written = [site_dir / "index.html"]
    (site_dir / "index.html").write_text(render_index(days), encoding="utf-8")
    (site_dir / "robots.txt").write_text(ROBOTS, encoding="utf-8")
    written.append(site_dir / "robots.txt")

    # `days` is newest first, so the previous entry is the newer neighbour.
    for i, day in enumerate(days):
        newer = days[i - 1]["date"] if i > 0 else None
        older = days[i + 1]["date"] if i + 1 < len(days) else None
        path = site_dir / f"{day['date']}.html"
        path.write_text(
            render_day(day, newer=newer, older=older, blurb=blurb), encoding="utf-8"
        )
        written.append(path)
    return written

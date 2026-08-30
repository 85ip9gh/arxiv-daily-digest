"""Contrary Research deep-dive client, read from the public Prismic content API.

Mirrors `hackernews.py`'s shape: same `Fetched`-style wrapper, same retry
contract, the same "picked, not summarized" contribution to the digest. The
two lighter sources are read the same way so the rest of the pipeline does not
need to know which one it is looking at.

The one real difference is cadence. Hacker News is a firehose narrowed by an
hour window; Contrary publishes a deep dive on the order of once a week, and
there are only a few dozen in total. So there is no hour window to widen. The
whole set of editorial deep dives comes back in one request, newest first by
publish date, and the selector picks a couple from it. Recency is a fixed-size
newest-first slice, not a time window.

Contrary's site is a Prismic-backed SPA. The reader-facing `article` documents
split into company business breakdowns (the bulk of them) and editorial deep
dives, told apart by the `deepDive` flag. Both are fetched: every deep dive,
and the newest slice of company breakdowns, whose recent entries are mostly AI,
tech and the business around them. Two searches, merged newest first, each row
tagged with its `kind` so the rest of the pipeline can show which it is.
"""
from __future__ import annotations

import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from .arxiv import USER_AGENT

API_ROOT = "https://contrary-research.cdn.prismic.io/api/v2"
SITE = "https://research.contrary.com"
DEFAULT_TIMEOUT = 30

# Every article kind resolves under /report/<uid>, company breakdowns and deep
# dives alike, so the reader link is built from the uid without caring which
# kind it is. /deep-dive/<uid> also exists but 308-redirects here.
REPORT_PATH = "/report/"

# Prismic caps a page at 100 and there are well under that many deep dives, so
# one page holds the whole set and the code can sort it in memory rather than
# trusting an ordering over a custom field.
PAGE_SIZE = 100

# Contrary's preview text is often a templated SEO string with no real content:
# the deep-dive variants are the bare sentence, and every company breakdown opens
# "A report from Contrary Research. Discover <Company>'s founding story...". A
# prefix match drops both, where an exact match caught only the deep-dive form.
_BOILERPLATE_PREFIXES = (
    "a report from contrary research",
    "a deep dive from contrary research",
    "a perspective from contrary research",
    "a transcript from contrary research",
)


class FetchError(RuntimeError):
    """Contrary was unreachable, or the API returned something that was not JSON."""


@dataclass(frozen=True)
class Article:
    """One Contrary article, flattened to the fields the digest uses.

    `kind` is "deep dive" or "company breakdown", from the `deepDive` flag. It
    defaults to "deep dive" so an article parsed from a payload that never
    carried the flag (an older archive, a test fixture) reads as one, which is
    what the whole catalogue was before company breakdowns were folded in.
    """

    article_id: str
    title: str
    url: str
    published: datetime
    authors: tuple[str, ...] = ()
    description: str = ""
    kind: str = "deep dive"

    @property
    def author_line(self) -> str:
        return ", ".join(self.authors)


def _parse_dt(raw: object) -> datetime | None:
    """Parse Prismic's `2026-08-21T00:07:00+0000` timestamps.

    strptime with an explicit `%z` reads the colon-less offset on every
    supported Python; fromisoformat is a lenient fallback for anything else.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(raw.strip())
    except ValueError:
        return None


def _rich_text(value: object) -> str:
    """The plain text of a Prismic rich-text field: `[{type, text, ...}, ...]`."""
    if not isinstance(value, list):
        return ""
    return " ".join(
        block.get("text", "").strip()
        for block in value
        if isinstance(block, dict) and block.get("text")
    ).strip()


def _title_case_slug(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.replace("_", "-").split("-") if part)


def _authors(value: object) -> tuple[str, ...]:
    """Author display names, from the linked author document where the query
    embedded it, falling back to a title-cased slug when it did not.

    The name lives in the author document's own `author` field. `fetchLinks`
    embeds it under the link's `data`; without the embed only the slug is here,
    and `jack-marks` reads well enough as `Jack Marks` for a byline.
    """
    if not isinstance(value, list):
        return ()
    names = []
    for entry in value:
        link = entry.get("author") if isinstance(entry, dict) else None
        if not isinstance(link, dict):
            continue
        data = link.get("data")
        name = ""
        if isinstance(data, dict):
            name = str(data.get("author") or "").strip()
        if not name and link.get("slug"):
            name = _title_case_slug(str(link["slug"]))
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _description(value: object) -> str:
    text = str(value or "").strip()
    low = text.lower()
    if any(low.startswith(prefix) for prefix in _BOILERPLATE_PREFIXES):
        return ""
    return text


def _parse_result(result: dict) -> Article | None:
    if not isinstance(result, dict):
        return None
    uid = str(result.get("uid") or "").strip()
    if not uid:
        return None
    data = result.get("data")
    if not isinstance(data, dict):
        return None
    title = _rich_text(data.get("title"))
    if not title:
        preview = str(data.get("previewTitle") or "").split("|")[0].strip()
        title = preview
    if not title:
        return None
    published = _parse_dt(data.get("datePublished")) or _parse_dt(
        result.get("first_publication_date")
    )
    if published is None:
        published = datetime.now(timezone.utc)
    # Only an explicit false marks a company breakdown. A missing flag means the
    # payload never carried it, and the safe reading is a deep dive.
    kind = "company breakdown" if data.get("deepDive") is False else "deep dive"
    return Article(
        article_id=uid,
        title=title,
        url=f"{SITE}{REPORT_PATH}{uid}",
        published=published,
        authors=_authors(data.get("authors")),
        description=_description(data.get("previewDescription")),
        kind=kind,
    )


def parse_results(payload: dict) -> list[Article]:
    """Turn a Prismic search response into articles, skipping unusable ones and
    deduplicating by uid, newest first by publish date.

    The deep-dive and company-breakdown searches are merged into one payload
    before parsing. A report is one kind or the other, never both, so the dedup
    is a guard on that invariant rather than something the data needs, and the
    first mention of a uid (the deep-dive search runs first) wins.
    """
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    articles = [a for a in (_parse_result(r) for r in results) if a is not None]
    seen: set[str] = set()
    unique: list[Article] = []
    for article in articles:
        if article.article_id in seen:
            continue
        seen.add(article.article_id)
        unique.append(article)
    unique.sort(key=lambda a: a.published, reverse=True)
    return unique


def _get(url: str, *, timeout: int, retries: int) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                url, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
            continue
        try:
            return response.json()
        except ValueError as exc:
            raise FetchError(f"Contrary returned unparseable JSON: {exc}") from exc
    raise FetchError(f"Contrary fetch failed after {retries + 1} tries: {last_error}")


def _master_ref(payload: dict) -> str:
    for ref in payload.get("refs", []):
        if isinstance(ref, dict) and ref.get("id") == "master":
            return str(ref.get("ref") or "")
    raise FetchError("Contrary API returned no master ref")


def build_query(ref: str) -> str:
    """The search URL for published editorial deep dives, newest first.

    `fetchLinks` pulls each linked author's name into the response so a byline
    needs no extra request. The predicate filters to the `deepDive` flag so the
    several hundred company breakdowns never enter the candidate pool.
    """
    params = {
        "ref": ref,
        "q": '[[at(document.type,"article")][at(my.article.deepDive,true)]]',
        "pageSize": PAGE_SIZE,
        "orderings": "[document.first_publication_date desc]",
        "fetchLinks": "author.author",
    }
    return f"{API_ROOT}/documents/search?{urllib.parse.urlencode(params)}"


# Company breakdowns are the bulk of the catalogue (462 against ~60 deep dives)
# and the recent ones are overwhelmingly AI, tech and the business around them,
# which is what a reader of this digest wants. Only the newest slice is fetched,
# a fixed count, not a time window: enough to keep the candidate pool current
# without pulling years of history no one will pick. `datePublished` is the
# reliable recency field here, the mirror image of the deep dives: a breakdown's
# `first_publication_date` is frequently a migration timestamp years off its real
# date, while its `datePublished` is sound.
COMPANY_PAGE_SIZE = 24

# Trim each result to the fields the digest reads. Without it a page of company
# breakdowns is ~8.6 MB of body content; with it the same page is a few tens of
# kilobytes. `deepDive` is kept so the parser can still tag the kind.
_COMPANY_FETCH = (
    "article.title,article.previewTitle,article.datePublished,"
    "article.previewDescription,article.authors,article.deepDive"
)


def build_company_query(ref: str) -> str:
    """The search URL for the newest company breakdowns, newest first by publish
    date.

    The predicate is the complement of `build_query`'s: an article that is not
    flagged as a deep dive. Ordered by `datePublished` for the recency reason
    above, trimmed by `fetch` to keep the payload small, and `fetchLinks` still
    pulls each author's name so a byline needs no extra request.
    """
    params = {
        "ref": ref,
        "q": '[[at(document.type,"article")][not(my.article.deepDive,true)]]',
        "pageSize": COMPANY_PAGE_SIZE,
        "orderings": "[my.article.datePublished desc]",
        "fetch": _COMPANY_FETCH,
        "fetchLinks": "author.author",
    }
    return f"{API_ROOT}/documents/search?{urllib.parse.urlencode(params)}"


@dataclass(frozen=True)
class Fetched:
    """What came back. No window field: Contrary has no hour window to report."""

    articles: list[Article]

    def __len__(self) -> int:
        return len(self.articles)


def fetch_recent(
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = 2,
) -> Fetched:
    """Fetch the deep dives and the newest company breakdowns, merged and newest
    first by publish date.

    Three requests: the API root carries the master ref the searches pin to,
    then the two searches. The deep-dive search is the original source and its
    failure is fatal; the company-breakdown search is additive, so a failure
    there degrades to deep dives only rather than losing the day. Each retries
    on a network error the same way the arXiv and Hacker News fetches do.
    """
    root = _get(API_ROOT, timeout=timeout, retries=retries)
    ref = _master_ref(root)
    deep = _get(build_query(ref), timeout=timeout, retries=retries)
    results = list(deep.get("results") or [])
    try:
        company = _get(build_company_query(ref), timeout=timeout, retries=retries)
        results += list(company.get("results") or [])
    except FetchError:
        pass
    return Fetched(parse_results({"results": results}))

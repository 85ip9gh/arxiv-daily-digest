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
dives. The `deepDive` flag is what separates the essays a reader follows the
site for from the several hundred company profiles, so the query asks Prismic
for that flag directly rather than fetching everything and filtering here.
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

_BOILERPLATE = (
    "a report from contrary research.",
    "a deep dive from contrary research.",
    "a perspective from contrary research.",
    "a transcript from contrary research.",
)


class FetchError(RuntimeError):
    """Contrary was unreachable, or the API returned something that was not JSON."""


@dataclass(frozen=True)
class Article:
    """One Contrary deep dive, flattened to the fields the digest uses."""

    article_id: str
    title: str
    url: str
    published: datetime
    authors: tuple[str, ...] = ()
    description: str = ""

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
    if text.lower() in _BOILERPLATE:
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
    return Article(
        article_id=uid,
        title=title,
        url=f"{SITE}{REPORT_PATH}{uid}",
        published=published,
        authors=_authors(data.get("authors")),
        description=_description(data.get("previewDescription")),
    )


def parse_results(payload: dict) -> list[Article]:
    """Turn a Prismic search response into articles, skipping unusable ones,
    newest first by publish date."""
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    articles = [a for a in (_parse_result(r) for r in results) if a is not None]
    articles.sort(key=lambda a: a.published, reverse=True)
    return articles


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
    """Fetch the current editorial deep dives, newest first by publish date.

    Two requests, the standard Prismic flow: the API root carries the master
    ref that the search must be pinned to, then the search itself. Both retry
    on a network error the same way the arXiv and Hacker News fetches do.
    """
    root = _get(API_ROOT, timeout=timeout, retries=retries)
    ref = _master_ref(root)
    payload = _get(build_query(ref), timeout=timeout, retries=retries)
    return Fetched(parse_results(payload))

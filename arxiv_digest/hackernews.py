"""Hacker News Algolia search client.

Mirrors `arxiv.py`'s shape on purpose: same window-widening fetch, same
`Fetched`-style wrapper, same retry contract. The two sources are read the
same way so the rest of the pipeline does not need to know which one it is
looking at.
"""
from __future__ import annotations

import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

from .arxiv import USER_AGENT

API_URL = "https://hn.algolia.com/api/v1/search_by_date"
DEFAULT_TIMEOUT = 30


class FetchError(RuntimeError):
    """Hacker News was unreachable, or returned something that was not JSON."""


@dataclass(frozen=True)
class Story:
    """One HN story, flattened to the fields the digest actually uses.

    Self-text posts ("Ask HN", "Show HN" without a link) have no external
    `url` and are skipped at fetch time. Reading and summarizing a story's own
    text box is a real feature, just not this one's: out of scope for v1.
    """

    hn_id: str
    title: str
    url: str | None
    points: int
    num_comments: int
    author: str
    created: datetime

    @property
    def hn_url(self) -> str:
        return f"https://news.ycombinator.com/item?id={self.hn_id}"


def _parse_hit(hit: dict) -> Story | None:
    url = hit.get("url")
    if not url:
        return None
    hn_id = str(hit.get("objectID", "")).strip()
    if not hn_id:
        return None
    created_raw = hit.get("created_at_i")
    try:
        created = datetime.fromtimestamp(int(created_raw), tz=timezone.utc)
    except (TypeError, ValueError):
        return None
    return Story(
        hn_id=hn_id,
        title=str(hit.get("title") or "").strip(),
        url=url,
        points=int(hit.get("points") or 0),
        num_comments=int(hit.get("num_comments") or 0),
        author=str(hit.get("author") or "").strip(),
        created=created,
    )


def parse_hits(payload: dict) -> list[Story]:
    """Turn an Algolia search response into stories, skipping unusable hits."""
    hits = payload.get("hits") if isinstance(payload, dict) else None
    if not isinstance(hits, list):
        return []
    stories = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        story = _parse_hit(hit)
        if story is not None:
            stories.append(story)
    return stories


def build_query(since_unix: int, min_points: int, hits_per_page: int) -> str:
    params = {
        "tags": "story",
        "numericFilters": f"created_at_i>{since_unix},points>={min_points}",
        "hitsPerPage": hits_per_page,
    }
    return f"{API_URL}?{urllib.parse.urlencode(params)}"


@dataclass(frozen=True)
class Fetched:
    """What came back, and how far back the code had to reach to get it."""

    stories: list[Story]
    hours: int

    def __len__(self) -> int:
        return len(self.stories)


def fetch_recent(
    *,
    hours: int = 48,
    min_points: int = 60,
    max_results: int = 100,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = 2,
    min_results: int = 12,
    max_hours: int = 168,
) -> Fetched:
    """Fetch recent stories, newest first, widening the window until there are enough.

    One request, queried at `max_hours` so the widest window this call could
    ever want is already in hand, then the same doubling-window logic as
    `arxiv.fetch_recent` narrows or widens over that response in memory. A
    quiet stretch should reach back further rather than publish nothing, and
    since the feed already covers the widest window, widening costs no extra
    request, exactly the same reasoning as the arXiv side.
    """
    since_unix = int(datetime.now(timezone.utc).timestamp()) - max_hours * 3600
    url = build_query(since_unix, min_points, max_results)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                url, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
    else:
        raise FetchError(
            f"Hacker News fetch failed after {retries + 1} tries: {last_error}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise FetchError(f"Hacker News returned unparseable JSON: {exc}") from exc

    everything = parse_hits(payload)
    now = datetime.now(timezone.utc)
    window = hours
    while True:
        kept = [s for s in everything if s.created >= now - timedelta(hours=window)]
        if len(kept) >= min_results or window >= max_hours:
            return Fetched(kept, window)
        window = min(window * 2, max_hours)

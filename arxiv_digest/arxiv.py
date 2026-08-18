"""arXiv Atom API client.

One request per run. arXiv asks callers to leave three seconds between
requests, so the fetch is deliberately a single query with a wide
`max_results` rather than a loop over categories.
"""
from __future__ import annotations

import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

API_URL = "http://export.arxiv.org/api/query"
USER_AGENT = "arxiv-daily-digest/0.1 (https://github.com/85ip9gh/arxiv-daily-digest)"

DEFAULT_CATEGORIES = ("cs.AI", "cs.LG", "cs.CL", "cs.SE", "cs.DC")
DEFAULT_TIMEOUT = 60

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

_WHITESPACE = re.compile(r"\s+")


class FetchError(RuntimeError):
    """arXiv was unreachable, or returned something that was not a feed."""


@dataclass(frozen=True)
class Paper:
    """One arXiv entry, flattened to the fields the digest actually uses."""

    arxiv_id: str  # versionless, e.g. "2408.01234"
    version: str  # e.g. "v1"
    title: str
    authors: tuple[str, ...]
    abstract: str
    categories: tuple[str, ...]
    primary_category: str
    published: datetime
    abs_url: str
    pdf_url: str

    @property
    def author_line(self) -> str:
        if len(self.authors) <= 3:
            return ", ".join(self.authors)
        return f"{', '.join(self.authors[:3])} and {len(self.authors) - 3} others"


def _clean(text: str | None) -> str:
    """Collapse the newlines arXiv wraps titles and abstracts at."""
    return _WHITESPACE.sub(" ", (text or "")).strip()


def _parse_entry(entry: ET.Element) -> Paper | None:
    raw_id = _clean(entry.findtext(f"{ATOM}id"))
    match = re.search(r"abs/(.+?)(v(\d+))?$", raw_id)
    if not match:
        return None
    arxiv_id = match.group(1)
    version = match.group(2) or ""

    published_text = _clean(entry.findtext(f"{ATOM}published"))
    try:
        published = datetime.fromisoformat(published_text.replace("Z", "+00:00"))
    except ValueError:
        return None

    authors = tuple(
        _clean(node.findtext(f"{ATOM}name"))
        for node in entry.findall(f"{ATOM}author")
        if _clean(node.findtext(f"{ATOM}name"))
    )
    categories = tuple(
        node.attrib["term"]
        for node in entry.findall(f"{ATOM}category")
        if "term" in node.attrib
    )
    primary = entry.find(f"{ARXIV_NS}primary_category")
    primary_category = primary.attrib.get("term", "") if primary is not None else ""

    abs_url = raw_id.replace("http://", "https://")
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}{version}"
    for link in entry.findall(f"{ATOM}link"):
        if link.attrib.get("title") == "pdf" and link.attrib.get("href"):
            pdf_url = link.attrib["href"].replace("http://", "https://")

    return Paper(
        arxiv_id=arxiv_id,
        version=version,
        title=_clean(entry.findtext(f"{ATOM}title")),
        authors=authors,
        abstract=_clean(entry.findtext(f"{ATOM}summary")),
        categories=categories or ((primary_category,) if primary_category else ()),
        primary_category=primary_category or (categories[0] if categories else ""),
        published=published,
        abs_url=abs_url,
        pdf_url=pdf_url,
    )


def parse_feed(xml_text: str) -> list[Paper]:
    """Turn an arXiv Atom response into papers, skipping entries we cannot read."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise FetchError(f"arXiv returned unparseable XML: {exc}") from exc

    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        paper = _parse_entry(entry)
        if paper is not None:
            papers.append(paper)
    return papers


def build_query(categories: tuple[str, ...] | list[str], max_results: int) -> str:
    clause = " OR ".join(f"cat:{c}" for c in categories)
    params = {
        "search_query": clause,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": max_results,
    }
    return f"{API_URL}?{urllib.parse.urlencode(params)}"


@dataclass(frozen=True)
class Fetched:
    """What came back, and how far back the code had to reach to get it."""

    papers: list[Paper]
    hours: int

    def __len__(self) -> int:
        return len(self.papers)


def fetch_recent(
    *,
    categories: tuple[str, ...] | list[str] = DEFAULT_CATEGORIES,
    hours: int = 48,
    max_results: int = 120,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = 2,
    min_results: int = 12,
    max_hours: int = 168,
) -> Fetched:
    """Fetch recent papers, newest first, widening the window until there are enough.

    The window starts at 48 hours rather than 24 because arXiv announces on
    weekdays only. It widens because that is still not enough: a Saturday run
    measured zero papers inside 48 hours and 120 inside 72. A digest that
    silently produces nothing on a quiet weekend is worse than one that reaches
    back to Thursday and says so, and `seen.json` already stops repeats.

    Widening costs no extra requests. The feed is sorted by submission date, so
    a wider window is a later cutoff over the response already in hand.
    """
    url = build_query(tuple(categories), max_results)
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
        raise FetchError(f"arXiv fetch failed after {retries + 1} tries: {last_error}")

    everything = parse_feed(response.text)
    now = datetime.now(timezone.utc)
    window = hours
    while True:
        kept = [p for p in everything if p.published >= now - timedelta(hours=window)]
        if len(kept) >= min_results or window >= max_hours:
            return Fetched(kept, window)
        window = min(window * 2, max_hours)

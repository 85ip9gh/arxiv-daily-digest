from datetime import datetime, timezone
from pathlib import Path

from arxiv_digest.arxiv import build_query, parse_feed

FIXTURE = Path(__file__).parent / "fixtures" / "feed.xml"


def papers():
    return parse_feed(FIXTURE.read_text(encoding="utf-8"))


def test_entry_without_a_usable_date_is_skipped():
    assert len(papers()) == 2


def test_version_is_split_off_the_id():
    first = papers()[0]
    assert first.arxiv_id == "2508.01234"
    assert first.version == "v1"


def test_wrapped_title_and_abstract_are_collapsed():
    first = papers()[0]
    assert first.title == "A Small Model That Cites Its Sources"
    assert "\n" not in first.abstract
    assert first.abstract.startswith("We show that requiring")


def test_links_are_upgraded_to_https():
    first = papers()[0]
    assert first.abs_url == "https://arxiv.org/abs/2508.01234v1"
    assert first.pdf_url == "https://arxiv.org/pdf/2508.01234v1"


def test_metadata_is_carried_through():
    first, second = papers()
    assert first.authors == ("Ada Rivers", "Bo Chen")
    assert first.primary_category == "cs.CL"
    assert set(first.categories) == {"cs.CL", "cs.AI"}
    assert first.published == datetime(2026, 8, 14, 17, 2, 11, tzinfo=timezone.utc)
    assert second.author_line == "Cara Nolan"


def test_author_line_truncates_long_lists():
    first = papers()[0]
    long_list = first.__class__(**{**first.__dict__, "authors": tuple("ABCDE")})
    assert long_list.author_line == "A, B, C and 2 others"


def test_query_asks_for_newest_first():
    url = build_query(("cs.AI", "cs.LG"), 50)
    assert "cat%3Acs.AI+OR+cat%3Acs.LG" in url
    assert "sortBy=submittedDate" in url
    assert "sortOrder=descending" in url
    assert "max_results=50" in url

from datetime import datetime, timezone
from pathlib import Path

from arxiv_digest.arxiv import DEFAULT_CATEGORIES, build_query, parse_feed

FIXTURE = Path(__file__).parent / "fixtures" / "feed.xml"


def test_default_categories_cover_software_engineering_and_infra():
    """The digest favours a working engineer's reading, not raw ML novelty."""
    assert "cs.SE" in DEFAULT_CATEGORIES
    assert "cs.DC" in DEFAULT_CATEGORIES
    assert set(DEFAULT_CATEGORIES) == {"cs.AI", "cs.LG", "cs.CL", "cs.SE", "cs.DC"}


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


class TestWindowWidening:
    """A Saturday run measured zero papers inside 48 hours and 120 inside 72."""

    def _feed(self, monkeypatch, ages_hours):
        from datetime import datetime, timedelta, timezone

        from arxiv_digest import arxiv as module

        now = datetime.now(timezone.utc)
        made = [
            papers()[0].__class__(
                **{
                    **papers()[0].__dict__,
                    "arxiv_id": f"id{i}",
                    "published": now - timedelta(hours=age),
                }
            )
            for i, age in enumerate(ages_hours)
        ]

        class Response:
            status_code = 200
            text = "<feed/>"

            def raise_for_status(self):
                pass

        monkeypatch.setattr(module.requests, "get", lambda *a, **k: Response())
        monkeypatch.setattr(module, "parse_feed", lambda _: made)
        return module

    def test_a_full_window_is_left_alone(self, monkeypatch):
        module = self._feed(monkeypatch, [1] * 20)
        result = module.fetch_recent(hours=48, min_results=12)
        assert result.hours == 48
        assert len(result) == 20

    def test_a_thin_window_widens_until_it_has_enough(self, monkeypatch):
        module = self._feed(monkeypatch, [60] * 20)
        result = module.fetch_recent(hours=48, min_results=12)
        assert result.hours == 96
        assert len(result) == 20

    def test_widening_stops_at_the_ceiling(self, monkeypatch):
        module = self._feed(monkeypatch, [400] * 20)
        result = module.fetch_recent(hours=48, min_results=12, max_hours=168)
        assert result.hours == 168
        assert len(result) == 0


def test_query_asks_for_newest_first():
    url = build_query(("cs.AI", "cs.LG"), 50)
    assert "cat%3Acs.AI+OR+cat%3Acs.LG" in url
    assert "sortBy=submittedDate" in url
    assert "sortOrder=descending" in url
    assert "max_results=50" in url

from datetime import datetime, timedelta, timezone

from arxiv_digest import hackernews as module
from arxiv_digest.hackernews import build_query, parse_hits


def hit(
    hn_id="1",
    title="A story",
    url="https://example.com/a",
    points=100,
    num_comments=42,
    author="pg",
    created_at_i=1_700_000_000,
):
    return {
        "objectID": hn_id,
        "title": title,
        "url": url,
        "points": points,
        "num_comments": num_comments,
        "author": author,
        "created_at_i": created_at_i,
    }


class TestParseHits:
    def test_a_hit_with_a_url_is_kept(self):
        stories = parse_hits({"hits": [hit()]})
        assert len(stories) == 1
        assert stories[0].hn_id == "1"
        assert stories[0].title == "A story"
        assert stories[0].points == 100
        assert stories[0].num_comments == 42
        assert stories[0].author == "pg"

    def test_a_self_text_post_with_no_url_is_skipped(self):
        ask_hn = hit(url=None)
        assert parse_hits({"hits": [ask_hn]}) == []

    def test_a_hit_with_no_id_is_skipped(self):
        assert parse_hits({"hits": [hit(hn_id="")]}) == []

    def test_a_hit_with_an_unparseable_timestamp_is_skipped(self):
        broken = hit()
        broken["created_at_i"] = "not-a-number"
        assert parse_hits({"hits": [broken]}) == []

    def test_a_non_dict_hit_is_skipped(self):
        stories = parse_hits({"hits": ["garbage", None, hit()]})
        assert len(stories) == 1
        assert stories[0].hn_id == "1"

    def test_no_hits_key_is_not_an_error(self):
        assert parse_hits({}) == []
        assert parse_hits({"hits": "not-a-list"}) == []


class TestHnUrl:
    def test_hn_url_points_at_the_discussion(self):
        story = parse_hits({"hits": [hit(hn_id="99")]})[0]
        assert story.hn_url == "https://news.ycombinator.com/item?id=99"


def test_query_asks_for_stories_above_the_point_floor():
    url = build_query(1_700_000_000, 60, 100)
    assert "tags=story" in url
    assert "points%3E%3D60" in url
    assert "created_at_i%3E1700000000" in url
    assert "hitsPerPage=100" in url


class TestWindowWidening:
    """Mirrors arxiv.fetch_recent: a quiet stretch reaches back further rather
    than publishing nothing."""

    def _feed(self, monkeypatch, ages_hours):
        now = datetime.now(timezone.utc)
        hits = [
            hit(
                hn_id=str(i),
                created_at_i=int((now - timedelta(hours=age)).timestamp()),
            )
            for i, age in enumerate(ages_hours)
        ]

        class Response:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"hits": hits}

        monkeypatch.setattr(module.requests, "get", lambda *a, **k: Response())
        return module

    def test_a_full_window_is_left_alone(self, monkeypatch):
        mod = self._feed(monkeypatch, [1] * 20)
        result = mod.fetch_recent(hours=48, min_results=12)
        assert result.hours == 48
        assert len(result) == 20

    def test_a_thin_window_widens_until_it_has_enough(self, monkeypatch):
        mod = self._feed(monkeypatch, [60] * 20)
        result = mod.fetch_recent(hours=48, min_results=12)
        assert result.hours == 96
        assert len(result) == 20

    def test_widening_stops_at_the_ceiling(self, monkeypatch):
        mod = self._feed(monkeypatch, [400] * 20)
        result = mod.fetch_recent(hours=48, min_results=12, max_hours=168)
        assert result.hours == 168
        assert len(result) == 0


class TestFetchErrors:
    def test_a_network_failure_raises_after_retries(self, monkeypatch):
        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise module.requests.RequestException("no route")

        monkeypatch.setattr(module.requests, "get", boom)
        monkeypatch.setattr(module.time, "sleep", lambda *_: None)
        try:
            module.fetch_recent(retries=1)
            assert False, "expected FetchError"
        except module.FetchError:
            pass
        assert len(calls) == 2

    def test_unparseable_json_raises(self, monkeypatch):
        class Response:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                raise ValueError("not json")

        monkeypatch.setattr(module.requests, "get", lambda *a, **k: Response())
        try:
            module.fetch_recent()
            assert False, "expected FetchError"
        except module.FetchError:
            pass

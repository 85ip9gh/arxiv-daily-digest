import urllib.parse
from datetime import datetime, timezone

from arxiv_digest import contrary as module
from arxiv_digest.contrary import (
    Article,
    build_company_query,
    build_query,
    parse_results,
)


def result(
    uid="a-deep-dive",
    title="A Deep Dive",
    date_published="2026-08-06T14:28:06+0000",
    first_pub="2026-08-10T21:01:47+0000",
    description="A deep dive from Contrary Research.",
    authors=(("jane-doe", "Jane Doe"),),
    deep_dive=None,
):
    data = {
        "title": [{"type": "paragraph", "text": title, "spans": []}] if title else [],
        "datePublished": date_published,
        "previewDescription": description,
        "authors": [
            {"author": {"slug": slug, "data": {"author": name} if name else None}}
            for slug, name in authors
        ],
    }
    if deep_dive is not None:
        data["deepDive"] = deep_dive
    return {"uid": uid, "first_publication_date": first_pub, "data": data}


class TestParseResults:
    def test_a_result_with_a_uid_and_title_is_kept(self):
        articles = parse_results({"results": [result()]})
        assert len(articles) == 1
        a = articles[0]
        assert a.article_id == "a-deep-dive"
        assert a.title == "A Deep Dive"
        assert a.url == "https://research.contrary.com/report/a-deep-dive"
        assert a.published.date().isoformat() == "2026-08-06"

    def test_a_result_with_no_uid_is_skipped(self):
        assert parse_results({"results": [result(uid="")]}) == []

    def test_a_result_with_no_title_is_skipped(self):
        assert parse_results({"results": [result(title="")]}) == []

    def test_preview_title_is_the_fallback_when_rich_title_is_empty(self):
        r = result(title="")
        r["data"]["previewTitle"] = "Report: The Fallback | Contrary Research"
        articles = parse_results({"results": [r]})
        assert articles[0].title == "Report: The Fallback"

    def test_a_missing_publish_date_falls_back_to_first_publication(self):
        articles = parse_results({"results": [result(date_published=None)]})
        assert articles[0].published.date().isoformat() == "2026-08-10"

    def test_a_non_dict_result_is_skipped(self):
        articles = parse_results({"results": ["garbage", None, result()]})
        assert len(articles) == 1

    def test_no_results_key_is_not_an_error(self):
        assert parse_results({}) == []
        assert parse_results({"results": "not-a-list"}) == []

    def test_results_come_back_newest_first(self):
        older = result(uid="older", date_published="2026-01-01T00:00:00+0000")
        newer = result(uid="newer", date_published="2026-08-01T00:00:00+0000")
        articles = parse_results({"results": [older, newer]})
        assert [a.article_id for a in articles] == ["newer", "older"]


class TestAuthors:
    def test_the_linked_name_is_used_when_the_query_embedded_it(self):
        a = parse_results({"results": [result(authors=(("jane-doe", "Jane Doe"),))]})[0]
        assert a.authors == ("Jane Doe",)
        assert a.author_line == "Jane Doe"

    def test_the_slug_is_title_cased_when_no_name_was_embedded(self):
        a = parse_results({"results": [result(authors=(("kyle-tianshi", None),))]})[0]
        assert a.authors == ("Kyle Tianshi",)

    def test_repeated_authors_are_deduplicated(self):
        a = parse_results(
            {"results": [result(authors=(("jane-doe", "Jane Doe"), ("jane-doe", "Jane Doe")))]}
        )[0]
        assert a.authors == ("Jane Doe",)


class TestDescription:
    def test_boilerplate_preview_text_is_dropped(self):
        a = parse_results({"results": [result(description="A deep dive from Contrary Research.")]})[0]
        assert a.description == ""

    def test_a_real_preview_description_is_kept(self):
        a = parse_results({"results": [result(description="Why launch costs collapsed.")]})[0]
        assert a.description == "Why launch costs collapsed."

    def test_company_seo_boilerplate_prefix_is_dropped(self):
        seo = (
            "A report from Contrary Research. Discover Legora's founding story, "
            "product, business model, and an in-depth analysis of the business."
        )
        a = parse_results({"results": [result(description=seo)]})[0]
        assert a.description == ""


class TestKind:
    def test_a_true_flag_tags_a_deep_dive(self):
        a = parse_results({"results": [result(deep_dive=True)]})[0]
        assert a.kind == "deep dive"

    def test_a_false_flag_tags_a_company_breakdown(self):
        a = parse_results({"results": [result(deep_dive=False)]})[0]
        assert a.kind == "company breakdown"

    def test_a_missing_flag_defaults_to_deep_dive(self):
        a = parse_results({"results": [result()]})[0]
        assert a.kind == "deep dive"


def test_query_asks_for_published_editorial_deep_dives():
    url = build_query("master-ref-123")
    assert "ref=master-ref-123" in url
    assert "my.article.deepDive" in url
    assert "fetchLinks=author.author" in url
    assert "document.type" in url


def test_company_query_asks_for_non_deep_dives_newest_first():
    url = build_company_query("master-ref-123")
    decoded = urllib.parse.unquote_plus(url)
    assert "ref=master-ref-123" in url
    assert "not(my.article.deepDive,true)" in decoded
    assert "my.article.datePublished desc" in decoded
    assert "fetch=" in url
    assert "fetchLinks=author.author" in url


def _responses(monkeypatch, *, ref="ref-abc", results=None, fail_times=0):
    """Mock requests.get for the two-step Prismic flow: the API root returns a
    master ref, the search returns results."""
    calls = {"n": 0}

    class Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def fake_get(url, *a, **k):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise module.requests.RequestException("no route")
        if "documents/search" in url:
            return Response({"results": results or []})
        return Response({"refs": [{"id": "master", "ref": ref}]})

    monkeypatch.setattr(module.requests, "get", fake_get)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    return calls


class TestFetchRecent:
    def test_it_fetches_the_ref_then_the_search(self, monkeypatch):
        _responses(monkeypatch, results=[result(uid="x"), result(uid="y")])
        fetched = module.fetch_recent()
        assert len(fetched) == 2
        assert {a.article_id for a in fetched.articles} == {"x", "y"}

    def test_a_network_failure_raises_after_retries(self, monkeypatch):
        calls = _responses(monkeypatch, fail_times=99)
        try:
            module.fetch_recent(retries=1)
            assert False, "expected FetchError"
        except module.FetchError:
            pass
        assert calls["n"] == 2

    def test_a_missing_master_ref_raises(self, monkeypatch):
        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"refs": []}

        monkeypatch.setattr(module.requests, "get", lambda *a, **k: Response())
        try:
            module.fetch_recent()
            assert False, "expected FetchError"
        except module.FetchError:
            pass

    def test_unparseable_json_raises(self, monkeypatch):
        class Response:
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


def _merged_responses(monkeypatch, *, deep, company, company_fails=False):
    """Mock the three-step flow fetch_recent now runs: the API root for the ref,
    then the deep-dive search, then the company-breakdown search."""
    calls = {"n": 0, "searches": 0}

    class Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def fake_get(url, *a, **k):
        calls["n"] += 1
        if "documents/search" in url:
            calls["searches"] += 1
            if calls["searches"] == 1:
                return Response({"results": deep})
            if company_fails:
                raise module.requests.RequestException("company down")
            return Response({"results": company})
        return Response({"refs": [{"id": "master", "ref": "r"}]})

    monkeypatch.setattr(module.requests, "get", fake_get)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    return calls


class TestMergedFetch:
    def test_deep_dives_and_company_breakdowns_are_merged_newest_first(self, monkeypatch):
        _merged_responses(
            monkeypatch,
            deep=[result(uid="essay", date_published="2026-08-01T00:00:00+0000", deep_dive=True)],
            company=[result(uid="legora", date_published="2026-08-28T00:00:00+0000", deep_dive=False)],
        )
        fetched = module.fetch_recent()
        assert [a.article_id for a in fetched.articles] == ["legora", "essay"]
        assert {a.kind for a in fetched.articles} == {"deep dive", "company breakdown"}

    def test_a_company_failure_degrades_to_deep_dives(self, monkeypatch):
        _merged_responses(
            monkeypatch,
            deep=[result(uid="essay", deep_dive=True)],
            company=[],
            company_fails=True,
        )
        fetched = module.fetch_recent(retries=0)
        assert [a.article_id for a in fetched.articles] == ["essay"]

    def test_a_uid_in_both_searches_appears_once_as_the_deep_dive(self, monkeypatch):
        _merged_responses(
            monkeypatch,
            deep=[result(uid="dup", deep_dive=True)],
            company=[result(uid="dup", deep_dive=False)],
        )
        fetched = module.fetch_recent()
        assert [a.article_id for a in fetched.articles] == ["dup"]
        assert fetched.articles[0].kind == "deep dive"


class TestArticleUrl:
    def test_the_report_path_is_universal(self):
        a = Article(
            article_id="ships-america",
            title="Ships for America",
            url=f"{module.SITE}{module.REPORT_PATH}ships-america",
            published=datetime(2026, 6, 29, tzinfo=timezone.utc),
        )
        assert a.url == "https://research.contrary.com/report/ships-america"

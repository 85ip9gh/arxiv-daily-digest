"""The run either adds a day to the archive or leaves it alone.

These cover the two ways a morning used to end up empty: one paper failing took
the whole digest with it, and asking for more papers than the daily token budget
allows failed every paper the same way.
"""
from datetime import datetime, timezone

import pytest

from arxiv_digest import arxiv, cli, contrary, digest, hackernews
from arxiv_digest.agent import Summary
from arxiv_digest.arxiv import Paper
from arxiv_digest.contrary import Article
from arxiv_digest.hackernews import Story
from arxiv_digest.llm import LLMError, RateLimitExhausted


def paper(n: int) -> Paper:
    return Paper(
        arxiv_id=f"2608.1000{n}",
        version="v1",
        title=f"Paper {n}",
        authors=("Ada Rivers",),
        abstract="An abstract.",
        categories=("cs.CL",),
        primary_category="cs.CL",
        published=datetime(2026, 8, 16, tzinfo=timezone.utc),
        abs_url=f"https://arxiv.org/abs/2608.1000{n}v1",
        pdf_url=f"https://arxiv.org/pdf/2608.1000{n}v1",
    )


def summary_for(p: Paper) -> Summary:
    return Summary(
        paper=p,
        problem="Something did not work.",
        approach="It works now.",
        result="Numbers moved.",
        so_what="Someone cares.",
        limitations="Narrow.",
        method_details=("one fact",),
        numbers=("accuracy 90 percent",),
        quote="a fragment of at least eight words copied out",
        reason="picked",
        grounded=True,
        unverified_numbers=(),
        read_full_text=True,
    )


def story(n: int) -> Story:
    return Story(
        hn_id=f"90000{n}",
        title=f"Story {n}",
        url=f"https://example.com/{n}",
        points=100 + n,
        num_comments=10 + n,
        author="pg",
        created=datetime(2026, 8, 16, tzinfo=timezone.utc),
    )


def article(n: int) -> Article:
    return Article(
        article_id=f"deep-dive-{n}",
        title=f"Deep Dive {n}",
        url=f"https://research.contrary.com/report/deep-dive-{n}",
        published=datetime(2026, 8, 16, tzinfo=timezone.utc),
        authors=("Ada Rivers",),
    )


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """A run with arXiv, Hacker News, Contrary and the model replaced, writing
    into a temp archive. The two lighter sources fetch empty by default, so
    tests that only care about the arXiv path do not have to think about them;
    tests that do can override their `fetch_recent` and selector themselves.
    """
    papers = [paper(n) for n in range(1, 6)]
    monkeypatch.setenv("ARXIV_DIGEST_API_KEY", "test-key")
    monkeypatch.delenv("ARXIV_DIGEST_BACKEND", raising=False)
    monkeypatch.setattr(
        arxiv, "fetch_recent", lambda **kw: arxiv.Fetched(papers=papers, hours=48)
    )
    monkeypatch.setattr(
        cli.agent, "select", lambda cands, **kw: [(p, "picked") for p in cands[: kw["count"]]]
    )
    monkeypatch.setattr(
        hackernews, "fetch_recent", lambda **kw: hackernews.Fetched(stories=[], hours=48)
    )
    monkeypatch.setattr(
        contrary, "fetch_recent", lambda **kw: contrary.Fetched(articles=[])
    )
    return tmp_path, papers


def run(tmp_path, *extra):
    return cli.main(
        ["--out-dir", str(tmp_path), "--no-site", "-n", "3", *extra]
    )


class TestPartialPublish:
    def test_a_clean_run_writes_every_paper(self, wired, monkeypatch):
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        assert run(tmp_path) == 0
        assert len(digest.load_days(tmp_path)[0]["papers"]) == 3

    def test_one_bad_paper_does_not_take_the_day_with_it(self, wired, monkeypatch, capsys):
        tmp_path, papers = wired

        def flaky(p, **kw):
            if p.arxiv_id == papers[1].arxiv_id:
                raise LLMError("model returned non-JSON")
            return summary_for(p)

        monkeypatch.setattr(cli.agent, "summarize", flaky)
        assert run(tmp_path) == 0
        written = digest.load_days(tmp_path)[0]["papers"]
        assert len(written) == 2
        assert papers[1].arxiv_id not in [r["arxiv_id"] for r in written]
        assert "publishing 2 of 3" in capsys.readouterr().err

    def test_running_out_of_budget_keeps_what_came_before_it(self, wired, monkeypatch):
        tmp_path, papers = wired
        done = []

        def until_broke(p, **kw):
            if len(done) >= 2:
                raise RateLimitExhausted("tokens per day (TPD): Limit 100000")
            done.append(p)
            return summary_for(p)

        monkeypatch.setattr(cli.agent, "summarize", until_broke)
        assert run(tmp_path) == 0
        assert len(digest.load_days(tmp_path)[0]["papers"]) == 2

    def test_budget_exhaustion_stops_asking_for_more(self, wired, monkeypatch):
        """Every remaining paper would hit the same wall, so it must not try them."""
        tmp_path, _ = wired
        calls = []

        def broke(p, **kw):
            calls.append(p)
            raise RateLimitExhausted("tokens per day (TPD): Limit 100000")

        monkeypatch.setattr(cli.agent, "summarize", broke)
        assert run(tmp_path) == 1
        assert len(calls) == 1

    def test_nothing_summarized_leaves_the_archive_untouched(self, wired, monkeypatch):
        tmp_path, _ = wired
        monkeypatch.setattr(
            cli.agent, "summarize", lambda p, **kw: (_ for _ in ()).throw(LLMError("no"))
        )
        assert run(tmp_path) == 1
        assert digest.load_days(tmp_path) == []


class TestCountCap:
    def test_more_than_the_cap_is_clamped(self, wired, monkeypatch, capsys):
        tmp_path, _ = wired
        seen = {}
        monkeypatch.setattr(
            cli.agent,
            "select",
            lambda cands, **kw: seen.setdefault("count", kw["count"]) and None
            or [(p, "picked") for p in cands[: kw["count"]]],
        )
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        cli.main(["--out-dir", str(tmp_path), "--no-site", "-n", "15"])
        assert seen["count"] == cli.MAX_COUNT
        assert "capping at 10" in capsys.readouterr().err

    def test_a_count_under_the_cap_is_left_alone(self, wired, monkeypatch):
        tmp_path, _ = wired
        seen = {}

        def spy(cands, **kw):
            seen["count"] = kw["count"]
            return [(p, "picked") for p in cands[: kw["count"]]]

        monkeypatch.setattr(cli.agent, "select", spy)
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        cli.main(["--out-dir", str(tmp_path), "--no-site", "-n", "3"])
        assert seen["count"] == 3


class TestAppend:
    """Topping a day up must not delete what the morning run produced."""

    def test_without_append_a_second_run_replaces_the_day(self, wired, monkeypatch):
        tmp_path, papers = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        cli.main(["--out-dir", str(tmp_path), "--no-site", "-n", "2"])
        first = [r["arxiv_id"] for r in digest.load_days(tmp_path)[0]["papers"]]

        # seen.json now filters the first run's papers out of the candidate pool.
        cli.main(["--out-dir", str(tmp_path), "--no-site", "-n", "2"])
        second = [r["arxiv_id"] for r in digest.load_days(tmp_path)[0]["papers"]]
        assert set(first).isdisjoint(second)
        assert len(second) == 2

    def test_append_keeps_the_earlier_papers(self, wired, monkeypatch):
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        cli.main(["--out-dir", str(tmp_path), "--no-site", "-n", "2"])
        first = [r["arxiv_id"] for r in digest.load_days(tmp_path)[0]["papers"]]

        cli.main(["--out-dir", str(tmp_path), "--no-site", "--append", "-n", "2"])
        after = [r["arxiv_id"] for r in digest.load_days(tmp_path)[0]["papers"]]
        assert after[:2] == first, "the earlier papers keep their positions"
        assert len(after) == 4

    def test_append_survives_a_run_that_dies_partway(self, wired, monkeypatch):
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        cli.main(["--out-dir", str(tmp_path), "--no-site", "-n", "2"])
        first = [r["arxiv_id"] for r in digest.load_days(tmp_path)[0]["papers"]]

        def broke(p, **kw):
            raise RateLimitExhausted("tokens per day (TPD): Limit 100000")

        monkeypatch.setattr(cli.agent, "summarize", broke)
        assert cli.main(["--out-dir", str(tmp_path), "--no-site", "--append", "-n", "2"]) == 1
        assert [r["arxiv_id"] for r in digest.load_days(tmp_path)[0]["papers"]] == first

    def test_append_on_an_empty_day_is_an_ordinary_run(self, wired, monkeypatch):
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        assert cli.main(["--out-dir", str(tmp_path), "--no-site", "--append", "-n", "2"]) == 0
        assert len(digest.load_days(tmp_path)[0]["papers"]) == 2

    def test_the_markdown_covers_the_whole_day_not_just_the_new_papers(
        self, wired, monkeypatch
    ):
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        cli.main(["--out-dir", str(tmp_path), "--no-site", "-n", "2"])
        cli.main(["--out-dir", str(tmp_path), "--no-site", "--append", "-n", "2"])
        from datetime import date

        text = (tmp_path / f"{date.today().isoformat()}.md").read_text(encoding="utf-8")
        assert "4 papers summarized, by" in text
        assert "## 4." in text


class TestHackerNews:
    """Hacker News is fetched and picked, never summarized or verified."""

    def _stories(self, n=3):
        return [story(i) for i in range(1, n + 1)]

    def test_a_run_publishes_stories_alongside_papers(self, wired, monkeypatch):
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        monkeypatch.setattr(
            hackernews, "fetch_recent", lambda **kw: hackernews.Fetched(stories=self._stories(), hours=48)
        )
        monkeypatch.setattr(
            cli.agent,
            "select_stories",
            lambda cands, **kw: [(s, "worth a click") for s in cands[: kw["count"]]],
        )
        assert run(tmp_path) == 0
        day = digest.load_days(tmp_path)[0]
        assert len(day["papers"]) == 3
        assert len(day["stories"]) == 3
        assert day["stories"][0]["reason"] == "worth a click"

    def test_no_hn_skips_the_second_source_entirely(self, wired, monkeypatch):
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))

        def explode(**kw):
            raise AssertionError("hackernews.fetch_recent must not be called")

        monkeypatch.setattr(hackernews, "fetch_recent", explode)
        assert run(tmp_path, "--no-hn") == 0
        assert digest.load_days(tmp_path)[0]["stories"] == []

    def test_hn_count_is_capped(self, wired, monkeypatch, capsys):
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        monkeypatch.setattr(
            hackernews,
            "fetch_recent",
            lambda **kw: hackernews.Fetched(stories=self._stories(20), hours=48),
        )
        seen = {}
        monkeypatch.setattr(
            cli.agent,
            "select_stories",
            lambda cands, **kw: seen.setdefault("count", kw["count"]) and None
            or [(s, "x") for s in cands[: kw["count"]]],
        )
        run(tmp_path, "--hn-count", "20")
        assert seen["count"] == cli.HN_MAX_COUNT
        assert f"capping at {cli.HN_MAX_COUNT}" in capsys.readouterr().err

    def test_hn_dedup_uses_its_own_seen_file(self, wired, monkeypatch):
        """seen-hn.json must be a different file from seen.json, or a paper's
        arxiv_id colliding with a story's hn_id would wrongly filter it out."""
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        monkeypatch.setattr(
            hackernews, "fetch_recent", lambda **kw: hackernews.Fetched(stories=self._stories(2), hours=48)
        )
        monkeypatch.setattr(
            cli.agent,
            "select_stories",
            lambda cands, **kw: [(s, "x") for s in cands[: kw["count"]]],
        )
        run(tmp_path)
        assert (tmp_path / digest.SEEN_HN_FILE).exists()
        assert (tmp_path / digest.SEEN_FILE).exists()
        hn_seen = digest.load_seen(tmp_path, filename=digest.SEEN_HN_FILE)
        assert hn_seen == {"900001", "900002"}

    def test_hn_repeats_skips_the_dedup_filter(self, wired, monkeypatch):
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        monkeypatch.setattr(
            hackernews, "fetch_recent", lambda **kw: hackernews.Fetched(stories=self._stories(1), hours=48)
        )
        picked = []
        monkeypatch.setattr(
            cli.agent,
            "select_stories",
            lambda cands, **kw: picked.append(len(cands)) or [(s, "x") for s in cands[: kw["count"]]],
        )
        run(tmp_path)
        run(tmp_path, "--hn-repeats")
        # Without --hn-repeats the second run would see zero candidates, since
        # the one story was already recorded in seen-hn.json by the first run.
        assert picked[-1] == 1


class TestContrary:
    """Contrary Research is fetched and picked, never summarized or verified,
    exactly like Hacker News."""

    def _articles(self, n=3):
        return [article(i) for i in range(1, n + 1)]

    def test_a_run_publishes_deep_dives_alongside_papers(self, wired, monkeypatch):
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        monkeypatch.setattr(
            contrary, "fetch_recent", lambda **kw: contrary.Fetched(articles=self._articles())
        )
        monkeypatch.setattr(
            cli.agent,
            "select_articles",
            lambda cands, **kw: [(a, "worth reading") for a in cands[: kw["count"]]],
        )
        assert run(tmp_path) == 0
        day = digest.load_days(tmp_path)[0]
        assert len(day["papers"]) == 3
        # --contrary-count defaults to 2, so two of the three candidates are picked.
        assert len(day["articles"]) == 2
        assert day["articles"][0]["reason"] == "worth reading"

    def test_no_contrary_skips_the_third_source_entirely(self, wired, monkeypatch):
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))

        def explode(**kw):
            raise AssertionError("contrary.fetch_recent must not be called")

        monkeypatch.setattr(contrary, "fetch_recent", explode)
        assert run(tmp_path, "--no-contrary") == 0
        assert digest.load_days(tmp_path)[0]["articles"] == []

    def test_contrary_count_is_capped(self, wired, monkeypatch, capsys):
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        monkeypatch.setattr(
            contrary,
            "fetch_recent",
            lambda **kw: contrary.Fetched(articles=self._articles(20)),
        )
        seen = {}
        monkeypatch.setattr(
            cli.agent,
            "select_articles",
            lambda cands, **kw: seen.setdefault("count", kw["count"]) and None
            or [(a, "x") for a in cands[: kw["count"]]],
        )
        run(tmp_path, "--contrary-count", "20")
        assert seen["count"] == cli.CONTRARY_MAX_COUNT
        assert f"capping at {cli.CONTRARY_MAX_COUNT}" in capsys.readouterr().err

    def test_contrary_dedup_uses_its_own_seen_file(self, wired, monkeypatch):
        """seen-contrary.json must be its own file, or an article_id colliding
        with a paper's arxiv_id or a story's hn_id would wrongly filter it out."""
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        monkeypatch.setattr(
            contrary, "fetch_recent", lambda **kw: contrary.Fetched(articles=self._articles(2))
        )
        monkeypatch.setattr(
            cli.agent,
            "select_articles",
            lambda cands, **kw: [(a, "x") for a in cands[: kw["count"]]],
        )
        run(tmp_path)
        assert (tmp_path / digest.SEEN_CONTRARY_FILE).exists()
        contrary_seen = digest.load_seen(tmp_path, filename=digest.SEEN_CONTRARY_FILE)
        assert contrary_seen == {"deep-dive-1", "deep-dive-2"}

    def test_contrary_repeats_skips_the_dedup_filter(self, wired, monkeypatch):
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))
        monkeypatch.setattr(
            contrary, "fetch_recent", lambda **kw: contrary.Fetched(articles=self._articles(1))
        )
        picked = []
        monkeypatch.setattr(
            cli.agent,
            "select_articles",
            lambda cands, **kw: picked.append(len(cands)) or [(a, "x") for a in cands[: kw["count"]]],
        )
        run(tmp_path)
        run(tmp_path, "--contrary-repeats")
        # Without --contrary-repeats the second run would see zero candidates,
        # the one deep dive having been recorded in seen-contrary.json already.
        assert picked[-1] == 1


class TestIndependentSourceFailure:
    """No source can take another down."""

    def test_arxiv_failure_still_lets_hn_publish(self, wired, monkeypatch, capsys):
        tmp_path, _ = wired

        def broke_fetch(**kw):
            raise arxiv.FetchError("arXiv is down")

        monkeypatch.setattr(arxiv, "fetch_recent", broke_fetch)
        monkeypatch.setattr(
            hackernews,
            "fetch_recent",
            lambda **kw: hackernews.Fetched(stories=[story(1)], hours=48),
        )
        monkeypatch.setattr(
            cli.agent,
            "select_stories",
            lambda cands, **kw: [(s, "worth a click") for s in cands[: kw["count"]]],
        )
        assert run(tmp_path) == 0
        day = digest.load_days(tmp_path)[0]
        assert day["papers"] == []
        assert len(day["stories"]) == 1
        assert "arxiv: fetch failed" in capsys.readouterr().err

    def test_hn_failure_still_lets_arxiv_publish(self, wired, monkeypatch, capsys):
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))

        def broke_fetch(**kw):
            raise hackernews.FetchError("hn is down")

        monkeypatch.setattr(hackernews, "fetch_recent", broke_fetch)
        assert run(tmp_path) == 0
        day = digest.load_days(tmp_path)[0]
        assert len(day["papers"]) == 3
        assert day["stories"] == []
        assert "hn: fetch failed" in capsys.readouterr().err

    def test_contrary_failure_still_lets_arxiv_publish(self, wired, monkeypatch, capsys):
        tmp_path, _ = wired
        monkeypatch.setattr(cli.agent, "summarize", lambda p, **kw: summary_for(p))

        def broke_fetch(**kw):
            raise contrary.FetchError("contrary is down")

        monkeypatch.setattr(contrary, "fetch_recent", broke_fetch)
        assert run(tmp_path) == 0
        day = digest.load_days(tmp_path)[0]
        assert len(day["papers"]) == 3
        assert day["articles"] == []
        assert "contrary: fetch failed" in capsys.readouterr().err

    def test_contrary_alone_publishes_when_the_others_fail(self, wired, monkeypatch):
        tmp_path, _ = wired

        def broke_arxiv(**kw):
            raise arxiv.FetchError("arXiv is down")

        def broke_hn(**kw):
            raise hackernews.FetchError("hn is down")

        monkeypatch.setattr(arxiv, "fetch_recent", broke_arxiv)
        monkeypatch.setattr(hackernews, "fetch_recent", broke_hn)
        monkeypatch.setattr(
            contrary, "fetch_recent", lambda **kw: contrary.Fetched(articles=[article(1)])
        )
        monkeypatch.setattr(
            cli.agent,
            "select_articles",
            lambda cands, **kw: [(a, "worth reading") for a in cands[: kw["count"]]],
        )
        assert run(tmp_path) == 0
        day = digest.load_days(tmp_path)[0]
        assert day["papers"] == []
        assert day["stories"] == []
        assert len(day["articles"]) == 1

    def test_all_three_sources_failing_is_the_only_thing_that_exits_nonzero(
        self, wired, monkeypatch
    ):
        tmp_path, _ = wired

        def broke_arxiv(**kw):
            raise arxiv.FetchError("arXiv is down")

        def broke_hn(**kw):
            raise hackernews.FetchError("hn is down")

        def broke_contrary(**kw):
            raise contrary.FetchError("contrary is down")

        monkeypatch.setattr(arxiv, "fetch_recent", broke_arxiv)
        monkeypatch.setattr(hackernews, "fetch_recent", broke_hn)
        monkeypatch.setattr(contrary, "fetch_recent", broke_contrary)
        assert run(tmp_path) == 1
        assert digest.load_days(tmp_path) == []

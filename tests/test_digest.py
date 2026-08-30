from dataclasses import replace
from datetime import date, datetime, timezone

from arxiv_digest.agent import Summary
from arxiv_digest.arxiv import Paper
from arxiv_digest.contrary import Article
from arxiv_digest.digest import (
    SEEN_CONTRARY_FILE,
    SEEN_HN_FILE,
    day_records,
    load_days,
    load_seen,
    merge_records,
    render,
    save_day,
    save_seen,
    write_digest,
)
from arxiv_digest.hackernews import Story

LONG_DASHES = (chr(0x2014), chr(0x2013))


def story(n: int = 1) -> Story:
    return Story(
        hn_id=f"90000{n}",
        title=f"Story {n}",
        url=f"https://example.com/{n}",
        points=100 + n,
        num_comments=10 + n,
        author="pg",
        created=datetime(2026, 8, 15, tzinfo=timezone.utc),
    )


def deepdive(n: int = 1) -> Article:
    return Article(
        article_id=f"deep-dive-{n}",
        title=f"Deep Dive {n}",
        url=f"https://research.contrary.com/report/deep-dive-{n}",
        published=datetime(2026, 8, 6, tzinfo=timezone.utc),
        authors=("Ada Rivers", "Bo Chen"),
    )


def summary(n: int, *, grounded: bool = True) -> Summary:
    paper = Paper(
        arxiv_id=f"2508.0000{n}",
        version="v1",
        title=f"Paper {n}",
        authors=("Ada Rivers", "Bo Chen"),
        abstract="An abstract.",
        categories=("cs.CL",),
        primary_category="cs.CL",
        published=datetime(2026, 8, 14, tzinfo=timezone.utc),
        abs_url=f"https://arxiv.org/abs/2508.0000{n}v1",
        pdf_url=f"https://arxiv.org/pdf/2508.0000{n}v1",
    )
    return Summary(
        paper=paper,
        problem="Models invent fields.",
        approach="Check the quote.",
        result="Errors fall.",
        so_what="Pipelines stop lying.",
        limitations="Extraction only.",
        method_details=("LoRA rank 16", "400 documents"),
        numbers=("error rate 18 to 2 percent",),
        quote="the error rate falls" if grounded else "",
        reason="has numbers",
        grounded=grounded,
        unverified_numbers=() if grounded else ("91.7",),
        read_full_text=True,
    )


def test_render_includes_links_and_sections():
    text = render([summary(1)], day=date(2026, 8, 15), model_label="qwen3:8b (local)")
    assert "# arXiv AI digest, 2026-08-15" in text
    assert "qwen3:8b (local)" in text
    assert "## 1. Paper 1" in text
    assert "[pdf](https://arxiv.org/pdf/2508.00001v1)" in text
    assert "**Why it matters.** Pipelines stop lying." in text
    assert "> the error rate falls" in text
    assert "*Picked because:* has numbers" in text


def test_technical_sections_are_rendered():
    text = render([summary(1)], day=date(2026, 8, 15), model_label="m")
    assert "**Method details.**" in text
    assert "- LoRA rank 16" in text
    assert "**Numbers.**" in text
    assert "**Limitations.** Extraction only." in text
    assert "read: full text" in text


def test_ungrounded_summary_is_flagged_not_hidden():
    text = render([summary(1, grounded=False)], day=date(2026, 8, 15), model_label="m")
    assert "Citation check failed" in text
    assert "1 of 1 summaries failed the citation check" in text
    assert "Figures not found in the source: 91.7." in text
    assert "Models invent fields." in text


def test_all_grounded_digest_has_no_warning_footer():
    text = render([summary(1), summary(2)], day=date(2026, 8, 15), model_label="m")
    assert "citation check" not in text
    assert "not in the source" not in text


def test_no_long_dashes_in_rendered_output():
    """The house rule is absolute, so the renderer's own strings are checked."""
    text = render([summary(1, grounded=False)], day=date(2026, 8, 15), model_label="m")
    for dash in LONG_DASHES:
        assert dash not in text


def test_digest_is_written_under_the_dated_name(tmp_path):
    path = write_digest("body\n", out_dir=tmp_path / "digests", day=date(2026, 8, 15))
    assert path.name == "2026-08-15.md"
    assert path.read_text(encoding="utf-8") == "body\n"


def test_archived_day_carries_everything_the_site_needs(tmp_path):
    save_day(tmp_path, day=date(2026, 8, 15), model_label="m", summaries=[summary(1)])
    days = load_days(tmp_path)
    assert len(days) == 1
    assert days[0]["date"] == "2026-08-15"
    assert days[0]["model"] == "m"
    paper = days[0]["papers"][0]
    assert paper["title"] == "Paper 1"
    assert paper["author_line"] == "Ada Rivers, Bo Chen"
    assert paper["pdf_url"].endswith("2508.00001v1")
    assert paper["so_what"] == "Pipelines stop lying."
    assert paper["grounded"] is True
    assert paper["method_details"] == ["LoRA rank 16", "400 documents"]
    assert paper["numbers"] == ["error rate 18 to 2 percent"]
    assert paper["limitations"] == "Extraction only."
    assert paper["unverified_numbers"] == []
    assert paper["source_label"] == "full text"


def test_days_come_back_newest_first(tmp_path):
    for day in (date(2026, 8, 13), date(2026, 8, 15), date(2026, 8, 14)):
        save_day(tmp_path, day=day, model_label="m", summaries=[summary(1)])
    assert [d["date"] for d in load_days(tmp_path)] == [
        "2026-08-15",
        "2026-08-14",
        "2026-08-13",
    ]


def test_a_corrupt_archived_day_is_skipped_not_fatal(tmp_path):
    save_day(tmp_path, day=date(2026, 8, 15), model_label="m", summaries=[summary(1)])
    (tmp_path / "data" / "2026-08-14.json").write_text("{broken", encoding="utf-8")
    assert [d["date"] for d in load_days(tmp_path)] == ["2026-08-15"]


def test_no_archive_directory_is_not_an_error(tmp_path):
    assert load_days(tmp_path) == []


def test_seen_round_trips_and_starts_empty(tmp_path):
    assert load_seen(tmp_path) == set()
    save_seen(tmp_path, set(), ["2508.00001", "2508.00002"])
    assert load_seen(tmp_path) == {"2508.00001", "2508.00002"}


def test_seen_survives_a_corrupt_file(tmp_path):
    (tmp_path / "seen.json").write_text("{not json", encoding="utf-8")
    assert load_seen(tmp_path) == set()


def test_seen_is_capped(tmp_path):
    from arxiv_digest import digest as digest_module

    existing = {f"id{i}" for i in range(digest_module.SEEN_LIMIT + 50)}
    save_seen(tmp_path, existing, ["fresh"])
    saved = load_seen(tmp_path)
    assert len(saved) == digest_module.SEEN_LIMIT
    assert "fresh" in saved


class TestSeenHnFile:
    """A second, separate seen file so paper and story dedup never collide."""

    def test_hn_seen_is_a_different_file_from_paper_seen(self, tmp_path):
        save_seen(tmp_path, set(), ["900001"], filename=SEEN_HN_FILE)
        assert load_seen(tmp_path) == set()
        assert load_seen(tmp_path, filename=SEEN_HN_FILE) == {"900001"}
        assert (tmp_path / SEEN_HN_FILE).name == "seen-hn.json"


class TestSeenContraryFile:
    """A third seen file so paper, story and article dedup never collide."""

    def test_contrary_seen_is_its_own_file(self, tmp_path):
        save_seen(tmp_path, set(), ["deep-dive-1"], filename=SEEN_CONTRARY_FILE)
        assert load_seen(tmp_path) == set()
        assert load_seen(tmp_path, filename=SEEN_HN_FILE) == set()
        assert load_seen(tmp_path, filename=SEEN_CONTRARY_FILE) == {"deep-dive-1"}
        assert (tmp_path / SEEN_CONTRARY_FILE).name == "seen-contrary.json"


class TestStoriesArchive:
    def test_a_day_with_only_stories_still_loads(self, tmp_path):
        save_day(
            tmp_path,
            day=date(2026, 8, 15),
            model_label="m",
            summaries=[],
            stories=[(story(1), "worth a click")],
        )
        days = load_days(tmp_path)
        assert len(days) == 1
        assert days[0]["papers"] == []
        record = days[0]["stories"][0]
        assert record["hn_id"] == "900001"
        assert record["title"] == "Story 1"
        assert record["url"] == "https://example.com/1"
        assert record["hn_url"] == "https://news.ycombinator.com/item?id=900001"
        assert record["points"] == 101
        assert record["num_comments"] == 11
        assert record["reason"] == "worth a click"

    def test_a_day_with_papers_and_stories_carries_both(self, tmp_path):
        save_day(
            tmp_path,
            day=date(2026, 8, 15),
            model_label="m",
            summaries=[summary(1)],
            stories=[(story(1), "worth a click")],
        )
        days = load_days(tmp_path)
        assert len(days[0]["papers"]) == 1
        assert len(days[0]["stories"]) == 1

    def test_an_old_day_with_no_stories_key_still_loads(self, tmp_path):
        """Backward compatibility with archives written before Hacker News existed."""
        import json

        save_day(tmp_path, day=date(2026, 8, 15), model_label="m", summaries=[summary(1)])
        path = tmp_path / "data" / "2026-08-15.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["stories"]
        path.write_text(json.dumps(payload), encoding="utf-8")

        days = load_days(tmp_path)
        assert len(days) == 1
        assert days[0]["papers"][0]["title"] == "Paper 1"

    def test_appending_stories_keeps_the_earlier_ones_in_place(self, tmp_path):
        save_day(
            tmp_path,
            day=date(2026, 8, 15),
            model_label="m",
            summaries=[],
            stories=[(story(1), "first")],
        )
        save_day(
            tmp_path,
            day=date(2026, 8, 15),
            model_label="m",
            summaries=[],
            stories=[(story(2), "second")],
            append=True,
        )
        stories = load_days(tmp_path)[0]["stories"]
        assert [s["hn_id"] for s in stories] == ["900001", "900002"]

    def test_without_append_a_second_run_replaces_the_days_stories(self, tmp_path):
        save_day(
            tmp_path,
            day=date(2026, 8, 15),
            model_label="m",
            summaries=[],
            stories=[(story(1), "first")],
        )
        save_day(
            tmp_path,
            day=date(2026, 8, 15),
            model_label="m",
            summaries=[],
            stories=[(story(2), "second")],
        )
        stories = load_days(tmp_path)[0]["stories"]
        assert [s["hn_id"] for s in stories] == ["900002"]


class TestArticlesArchive:
    def test_a_day_with_only_articles_still_loads(self, tmp_path):
        save_day(
            tmp_path,
            day=date(2026, 8, 15),
            model_label="m",
            summaries=[],
            articles=[(deepdive(1), "worth reading")],
        )
        days = load_days(tmp_path)
        assert len(days) == 1
        assert days[0]["papers"] == []
        assert days[0]["stories"] == []
        record = days[0]["articles"][0]
        assert record["article_id"] == "deep-dive-1"
        assert record["title"] == "Deep Dive 1"
        assert record["url"] == "https://research.contrary.com/report/deep-dive-1"
        assert record["author_line"] == "Ada Rivers, Bo Chen"
        assert record["reason"] == "worth reading"

    def test_a_day_with_all_three_kinds_carries_them(self, tmp_path):
        save_day(
            tmp_path,
            day=date(2026, 8, 15),
            model_label="m",
            summaries=[summary(1)],
            stories=[(story(1), "worth a click")],
            articles=[(deepdive(1), "worth reading")],
        )
        days = load_days(tmp_path)
        assert len(days[0]["papers"]) == 1
        assert len(days[0]["stories"]) == 1
        assert len(days[0]["articles"]) == 1

    def test_an_old_day_with_no_articles_key_still_loads(self, tmp_path):
        """Backward compatibility with archives written before Contrary existed."""
        import json

        save_day(tmp_path, day=date(2026, 8, 15), model_label="m", summaries=[summary(1)])
        path = tmp_path / "data" / "2026-08-15.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["articles"]
        path.write_text(json.dumps(payload), encoding="utf-8")

        days = load_days(tmp_path)
        assert len(days) == 1
        assert days[0]["papers"][0]["title"] == "Paper 1"

    def test_appending_articles_keeps_the_earlier_ones_in_place(self, tmp_path):
        save_day(
            tmp_path,
            day=date(2026, 8, 15),
            model_label="m",
            summaries=[],
            articles=[(deepdive(1), "first")],
        )
        save_day(
            tmp_path,
            day=date(2026, 8, 15),
            model_label="m",
            summaries=[],
            articles=[(deepdive(2), "second")],
            append=True,
        )
        articles = load_days(tmp_path)[0]["articles"]
        assert [a["article_id"] for a in articles] == ["deep-dive-1", "deep-dive-2"]


class TestMergeRecordsIdKey:
    def test_default_id_key_matches_the_existing_paper_callers(self):
        merged = merge_records([{"arxiv_id": "a"}], [{"arxiv_id": "a"}, {"arxiv_id": "b"}])
        assert [r["arxiv_id"] for r in merged] == ["a", "b"]

    def test_an_alternate_id_key_works_for_stories(self):
        merged = merge_records(
            [{"hn_id": "1"}], [{"hn_id": "1"}, {"hn_id": "2"}], id_key="hn_id"
        )
        assert [r["hn_id"] for r in merged] == ["1", "2"]


class TestDayRecordsKey:
    def test_default_key_reads_papers(self, tmp_path):
        save_day(tmp_path, day=date(2026, 8, 15), model_label="m", summaries=[summary(1)])
        records = day_records(tmp_path, date(2026, 8, 15))
        assert records[0]["title"] == "Paper 1"

    def test_stories_key_reads_stories(self, tmp_path):
        save_day(
            tmp_path,
            day=date(2026, 8, 15),
            model_label="m",
            summaries=[],
            stories=[(story(1), "worth a click")],
        )
        records = day_records(tmp_path, date(2026, 8, 15), key="stories")
        assert records[0]["hn_id"] == "900001"


class TestRenderWithStories:
    def test_the_hacker_news_section_lists_title_points_and_reason(self):
        text = render(
            [summary(1)],
            day=date(2026, 8, 15),
            model_label="m",
            stories=[(story(1), "worth a click")],
        )
        assert "## From Hacker News" in text
        assert "[Story 1](https://example.com/1)" in text
        assert "101 points, 11 comments" in text
        assert "worth a click" in text
        assert "[Discussion](https://news.ycombinator.com/item?id=900001)" in text

    def test_the_summary_line_counts_both_kinds_honestly(self):
        text = render(
            [summary(1)],
            day=date(2026, 8, 15),
            model_label="m",
            stories=[(story(1), "worth a click"), (story(2), "also good")],
        )
        assert "1 paper summarized and 2 Hacker News stories picked, by m." in text

    def test_a_stories_only_day_omits_the_paper_summary_line_wording(self):
        text = render([], day=date(2026, 8, 15), model_label="m", stories=[(story(1), "x")])
        assert "1 Hacker News story picked, by m." in text

    def test_no_stories_renders_no_hacker_news_section(self):
        text = render([summary(1)], day=date(2026, 8, 15), model_label="m")
        assert "From Hacker News" not in text

    def test_no_long_dashes_in_the_hacker_news_section(self):
        text = render(
            [], day=date(2026, 8, 15), model_label="m", stories=[(story(1), "x")]
        )
        for dash in LONG_DASHES:
            assert dash not in text


class TestRenderWithArticles:
    def test_the_contrary_section_lists_title_byline_kind_and_reason(self):
        text = render(
            [summary(1)],
            day=date(2026, 8, 15),
            model_label="m",
            articles=[(deepdive(1), "sharp on AI infra")],
        )
        assert "## From Contrary Research" in text
        assert "[Deep Dive 1](https://research.contrary.com/report/deep-dive-1)" in text
        assert "Deep dive by Ada Rivers, Bo Chen" in text
        assert "sharp on AI infra" in text

    def test_a_company_breakdown_shows_its_kind_in_the_section(self):
        breakdown = replace(deepdive(1), kind="company breakdown")
        text = render(
            [], day=date(2026, 8, 15), model_label="m", articles=[(breakdown, "big raise")]
        )
        assert "Company breakdown by Ada Rivers, Bo Chen. big raise" in text

    def test_the_summary_line_counts_all_three_kinds(self):
        text = render(
            [summary(1)],
            day=date(2026, 8, 15),
            model_label="m",
            stories=[(story(1), "x")],
            articles=[(deepdive(1), "y"), (deepdive(2), "z")],
        )
        assert (
            "1 paper summarized, 1 Hacker News story picked, and 2 Contrary "
            "Research pieces picked, by m." in text
        )

    def test_an_articles_only_day_reads_naturally(self):
        text = render([], day=date(2026, 8, 15), model_label="m", articles=[(deepdive(1), "x")])
        assert "1 Contrary Research piece picked, by m." in text

    def test_no_articles_renders_no_contrary_section(self):
        text = render([summary(1)], day=date(2026, 8, 15), model_label="m")
        assert "From Contrary Research" not in text

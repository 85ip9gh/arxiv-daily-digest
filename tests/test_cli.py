"""The run either adds a day to the archive or leaves it alone.

These cover the two ways a morning used to end up empty: one paper failing took
the whole digest with it, and asking for more papers than the daily token budget
allows failed every paper the same way.
"""
from datetime import datetime, timezone

import pytest

from arxiv_digest import agent, arxiv, cli, digest
from arxiv_digest.agent import Summary
from arxiv_digest.arxiv import Paper
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


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """A run with arXiv and the model replaced, writing into a temp archive."""
    papers = [paper(n) for n in range(1, 6)]
    monkeypatch.setenv("ARXIV_DIGEST_API_KEY", "test-key")
    monkeypatch.delenv("ARXIV_DIGEST_BACKEND", raising=False)
    monkeypatch.setattr(
        arxiv, "fetch_recent", lambda **kw: arxiv.Fetched(papers=papers, hours=48)
    )
    monkeypatch.setattr(
        cli.agent, "select", lambda cands, **kw: [(p, "picked") for p in cands[: kw["count"]]]
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
        assert "4 papers, summarized by" in text
        assert "## 4." in text

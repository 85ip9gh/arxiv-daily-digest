from datetime import date, datetime, timezone

from arxiv_digest.agent import Summary
from arxiv_digest.arxiv import Paper
from arxiv_digest.digest import load_seen, render, save_seen, write_digest

LONG_DASHES = (chr(0x2014), chr(0x2013))


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
        quote="the error rate falls" if grounded else "",
        reason="has numbers",
        grounded=grounded,
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


def test_ungrounded_summary_is_flagged_not_hidden():
    text = render([summary(1, grounded=False)], day=date(2026, 8, 15), model_label="m")
    assert "Citation check failed" in text
    assert "1 of 1 summaries failed the citation check." in text
    assert "Models invent fields." in text


def test_all_grounded_digest_has_no_warning_footer():
    text = render([summary(1), summary(2)], day=date(2026, 8, 15), model_label="m")
    assert "citation check" not in text


def test_no_long_dashes_in_rendered_output():
    """The house rule is absolute, so the renderer's own strings are checked."""
    text = render([summary(1, grounded=False)], day=date(2026, 8, 15), model_label="m")
    for dash in LONG_DASHES:
        assert dash not in text


def test_digest_is_written_under_the_dated_name(tmp_path):
    path = write_digest("body\n", out_dir=tmp_path / "digests", day=date(2026, 8, 15))
    assert path.name == "2026-08-15.md"
    assert path.read_text(encoding="utf-8") == "body\n"


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

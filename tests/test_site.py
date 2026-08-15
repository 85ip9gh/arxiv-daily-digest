from pathlib import Path

from arxiv_digest import site


def day(iso: str, *, grounded: bool = True) -> dict:
    return {
        "date": iso,
        "model": "llama-3.3-70b-versatile (api.groq.com)",
        "generated_at": f"{iso}T11:02:00+00:00",
        "papers": [
            {
                "arxiv_id": "2508.00001",
                "title": "A Small Model That Cites Its Sources",
                "author_line": "Ada Rivers, Bo Chen",
                "primary_category": "cs.CL",
                "abs_url": "https://arxiv.org/abs/2508.00001v1",
                "pdf_url": "https://arxiv.org/pdf/2508.00001v1",
                "problem": "Models invent fields.",
                "approach": "Require a quote & check it.",
                "result": "Errors fall to 2 percent.",
                "so_what": "Pipelines stop lying.",
                "quote": "the error rate falls",
                "grounded": grounded,
                "reason": "has numbers",
            }
        ],
    }


class TestIndex:
    def test_lists_every_day_with_its_paper_titles(self):
        page = site.render_index([day("2026-08-15"), day("2026-08-14")])
        assert 'href="2026-08-15.html"' in page
        assert 'href="2026-08-14.html"' in page
        assert "August 15, 2026" in page
        assert "A Small Model That Cites Its Sources" in page

    def test_empty_archive_still_renders(self):
        page = site.render_index([])
        assert "No digests yet." in page
        assert "<html" in page


class TestDayPage:
    def test_carries_the_four_fields_and_the_quote(self):
        page = site.render_day(day("2026-08-15"))
        assert "<b>Problem.</b> Models invent fields." in page
        assert "<b>Why it matters.</b> Pipelines stop lying." in page
        assert "<blockquote>the error rate falls</blockquote>" in page
        assert 'href="https://arxiv.org/pdf/2508.00001v1"' in page
        assert "Picked because: has numbers" in page

    def test_ungrounded_summary_is_flagged_on_the_page(self):
        page = site.render_day(day("2026-08-15", grounded=False))
        assert "Citation check failed" in page
        assert "Models invent fields." in page

    def test_text_is_escaped_not_injected(self):
        payload = day("2026-08-15")
        payload["papers"][0]["title"] = '<script>alert("x")</script>'
        page = site.render_day(payload)
        assert "<script>" not in page
        assert "&lt;script&gt;" in page
        # The ampersand in another field must survive as an entity too.
        assert "Require a quote &amp; check it." in page

    def test_pager_points_both_ways(self):
        page = site.render_day(
            day("2026-08-14"), newer="2026-08-15", older="2026-08-13"
        )
        assert "Newer: August 15, 2026" in page
        assert "Older: August 13, 2026" in page

    def test_no_pager_on_a_lone_day(self):
        assert '<nav class="pager">' not in site.render_day(day("2026-08-15"))


class TestBuild:
    def test_writes_an_index_and_one_page_per_day(self, tmp_path: Path):
        written = site.build([day("2026-08-15"), day("2026-08-14")], tmp_path)
        assert len(written) == 4
        assert (tmp_path / "index.html").exists()
        assert (tmp_path / "2026-08-15.html").exists()
        assert (tmp_path / "2026-08-14.html").exists()

    def test_the_site_asks_not_to_be_indexed(self, tmp_path: Path):
        site.build([day("2026-08-15")], tmp_path)
        assert (tmp_path / "robots.txt").read_text(encoding="utf-8") == (
            "User-agent: *\nDisallow: /\n"
        )

    def test_newest_day_has_no_newer_link(self, tmp_path: Path):
        site.build([day("2026-08-15"), day("2026-08-14")], tmp_path)
        newest = (tmp_path / "2026-08-15.html").read_text(encoding="utf-8")
        oldest = (tmp_path / "2026-08-14.html").read_text(encoding="utf-8")
        assert "Newer:" not in newest
        assert "Older: August 14, 2026" in newest
        assert "Newer: August 15, 2026" in oldest
        assert "Older:" not in oldest

    def test_pages_are_self_contained(self, tmp_path: Path):
        site.build([day("2026-08-15")], tmp_path)
        page = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "<style>" in page
        # Nothing may be fetched at page load: no scripts, no external assets.
        assert "<script" not in page
        assert "http://" not in page
        for tag in ("<link ", "src="):
            assert tag not in page

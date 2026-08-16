import re
from pathlib import Path

from arxiv_digest import site


def day(iso: str, *, grounded: bool = True, stray: list | None = None) -> dict:
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
                "limitations": "Extraction only, not classification.",
                "method_details": ["LoRA rank 16", "400 documents"],
                "numbers": ["error rate 18 to 2 percent"],
                "quote": "the error rate falls",
                "grounded": grounded,
                "unverified_numbers": stray or [],
                "read_full_text": True,
                "source_label": "full text",
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

    def test_each_row_carries_a_searchable_index(self):
        page = site.render_index([day("2026-08-15")])
        row = re.search(r'data-text="([^"]+)"', page).group(1)
        assert "saturday" in row
        assert "cs.cl" in row
        assert "cites its sources" in row

    def test_filter_and_count_are_present(self):
        page = site.render_index([day("2026-08-15"), day("2026-08-14")])
        assert 'id="filter"' in page
        assert ">2 days<" in page

    def test_empty_archive_still_renders(self):
        page = site.render_index([])
        assert "No digests yet." in page
        assert "<html" in page


class TestDayPage:
    def test_carries_the_prose_fields_and_the_quote(self):
        page = site.render_day(day("2026-08-15"))
        assert "<h3>Problem</h3><p>Models invent fields.</p>" in page
        assert "<h3>Why it matters</h3><p>Pipelines stop lying.</p>" in page
        assert "<blockquote>the error rate falls" in page
        assert 'href="https://arxiv.org/pdf/2508.00001v1"' in page
        assert "Picked because: has numbers" in page

    def test_technical_detail_sits_behind_disclosures(self):
        page = site.render_day(day("2026-08-15"))
        assert "<summary>Method details</summary>" in page
        assert "<li>LoRA rank 16</li>" in page
        assert "<summary>Numbers</summary>" in page
        assert "<summary>Limitations</summary>" in page
        assert "Extraction only, not classification." in page

    def test_a_paper_with_no_details_renders_no_empty_disclosure(self):
        payload = day("2026-08-15")
        payload["papers"][0]["method_details"] = []
        payload["papers"][0]["numbers"] = []
        payload["papers"][0]["limitations"] = ""
        page = site.render_day(payload)
        assert "<details" not in page
        assert "Models invent fields." in page

    def test_checks_are_shown_as_chips(self):
        page = site.render_day(day("2026-08-15"))
        assert "quote verified" in page
        assert "figures checked" in page
        assert "read: full text" in page

    def test_failed_checks_are_flagged_not_hidden(self):
        page = site.render_day(day("2026-08-15", grounded=False, stray=["91.7", "8.2"]))
        assert "quote unverified" in page
        assert "2 figures not in source" in page
        assert "91.7, 8.2" in page
        assert "Models invent fields." in page

    def test_text_is_escaped_not_injected(self):
        payload = day("2026-08-15")
        payload["papers"][0]["title"] = '<script>alert("x")</script>'
        page = site.render_day(payload)
        assert "<script>alert" not in page
        assert "&lt;script&gt;" in page
        assert "Require a quote &amp; check it." in page

    def test_pager_points_both_ways(self):
        page = site.render_day(
            day("2026-08-14"), newer="2026-08-15", older="2026-08-13"
        )
        assert 'id="newer"' in page and "August 15, 2026" in page
        assert 'id="older"' in page and "August 13, 2026" in page

    def test_no_pager_on_a_lone_day(self):
        assert '<nav class="pager">' not in site.render_day(day("2026-08-15"))


class TestThemeTokens:
    """The classic unreadable-page bug is a color defined only inside a media
    or [data-theme] block, which never applies in the un-stamped default."""

    def test_every_token_has_a_bare_root_definition(self):
        bare = site.STYLE.split("@media", 1)[0]
        declared = set(re.findall(r"(--[a-z-]+):", bare))
        used = set(re.findall(r"var\((--[a-z-]+)", site.STYLE))
        assert used <= declared, used - declared

    def test_both_explicit_themes_redefine_the_same_tokens(self):
        dark_media = re.search(
            r"prefers-color-scheme: dark\).*?\n\s*\}\n\}", site.STYLE, re.S
        ).group(0)
        stamped = re.search(
            r'\:root\[data-theme="dark"\] \{.*?\n\}', site.STYLE, re.S
        ).group(0)
        assert set(re.findall(r"(--[a-z-]+):", dark_media)) == set(
            re.findall(r"(--[a-z-]+):", stamped)
        )

    def test_the_dark_media_query_yields_to_an_explicit_light_choice(self):
        assert ':root:not([data-theme="light"])' in site.STYLE

    def test_body_paints_its_own_background(self):
        assert re.search(r"body \{[^}]*background: var\(--ground\)", site.STYLE, re.S)


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
        assert 'id="newer"' not in newest
        assert 'id="older"' in newest
        assert 'id="newer"' in oldest
        assert 'id="older"' not in oldest

    def test_pages_fetch_nothing_at_load(self, tmp_path: Path):
        """Scripts and styles are inline. Anything external would be a request
        the server cannot serve and a promise the deployment does not keep."""
        site.build([day("2026-08-15")], tmp_path)
        for name in ("index.html", "2026-08-15.html"):
            page = (tmp_path / name).read_text(encoding="utf-8")
            assert "src=" not in page
            assert "<link " not in page
            assert "@import" not in page
            assert "fetch(" not in page
            # The only absolute URLs allowed are the arXiv links a reader clicks.
            for url in re.findall(r"https?://[^\s\"']+", page):
                assert url.startswith("https://arxiv.org/"), url

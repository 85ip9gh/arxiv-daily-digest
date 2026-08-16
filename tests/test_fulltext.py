from arxiv_digest import fulltext
from arxiv_digest.fulltext import Section, pack, parse_sections

HTML = """
<html><head><style>.x { color: red }</style></head><body>
<h1>A Small Model That Cites Its Sources</h1>
<div class="ltx_abstract"><p>We show that requiring a quote removes fabrication.</p></div>
<h2>1 Introduction</h2>
<p>Language models are widely used.
They are also confidently wrong.</p>
<h2>2 Method</h2>
<p>We fine-tune a <math alttext="7B">garbage latex noise</math> decoder with LoRA rank 16.</p>
<p>Training runs for 3 epochs on 12k examples.</p>
<h2>3 Experiments</h2>
<table><tr><td>HotpotQA</td><td>91.4</td><td>84.2</td></tr></table>
<h2>References</h2>
<p>[1] Someone et al. A paper. 2024.</p>
<h2>Appendix A</h2>
<p>Extra proofs nobody needs in a summary.</p>
</body></html>
"""


class TestParsing:
    def test_headings_split_the_document(self):
        headings = [s.heading for s in parse_sections(HTML)]
        assert "2 Method" in headings
        assert "3 Experiments" in headings

    def test_references_and_appendix_are_dropped(self):
        headings = [s.heading for s in parse_sections(HTML)]
        assert not any(h.startswith("References") for h in headings)
        assert not any(h.startswith("Appendix") for h in headings)

    def test_math_and_style_markup_never_reaches_the_model(self):
        text = "\n".join(s.text for s in parse_sections(HTML))
        assert "garbage latex noise" not in text
        assert "color: red" not in text

    def test_table_values_survive_because_the_numbers_live_there(self):
        text = "\n".join(s.text for s in parse_sections(HTML))
        assert "91.4" in text
        assert "84.2" in text

    def test_wrapped_lines_are_joined(self):
        text = "\n".join(s.text for s in parse_sections(HTML))
        assert "LoRA rank 16" in text

    def test_malformed_html_returns_what_it_can(self):
        sections = parse_sections("<h2>Method</h2><p>We train a model<p>unclosed")
        assert any("We train a model" in s.text for s in sections)


class TestPacking:
    def test_method_and_results_outrank_the_introduction(self):
        sections = [
            Section("1 Introduction", "intro " * 200),
            Section("2 Method", "method " * 200),
            Section("3 Results", "results " * 200),
        ]
        packed = pack(sections, budget=1800)
        assert "method" in packed
        assert "results" in packed
        assert "intro" not in packed

    def test_kept_sections_stay_in_document_order(self):
        sections = [
            Section("2 Method", "method text"),
            Section("3 Results", "results text"),
        ]
        packed = pack(sections, budget=4000)
        assert packed.index("method text") < packed.index("results text")

    def test_the_budget_is_respected(self):
        sections = [Section(f"{i} Method", "x" * 5000) for i in range(6)]
        assert len(pack(sections, budget=3000)) <= 3000

    def test_nothing_in_nothing_out(self):
        assert pack([], budget=1000) == ""


class TestFetch:
    def test_a_stub_page_is_treated_as_no_full_text(self, monkeypatch):
        class Response:
            status_code = 200
            headers = {"Content-Type": "text/html"}
            text = "<html><body><p>No HTML is available for this paper.</p></body></html>"

        monkeypatch.setattr(fulltext.requests, "get", lambda *a, **k: Response())
        assert fulltext.fetch(_paper()) is None

    def test_a_non_200_is_not_an_exception(self, monkeypatch):
        class Response:
            status_code = 404
            headers = {"Content-Type": "text/html"}
            text = ""

        monkeypatch.setattr(fulltext.requests, "get", lambda *a, **k: Response())
        assert fulltext.fetch(_paper()) is None

    def test_a_network_failure_falls_back_rather_than_raising(self, monkeypatch):
        def boom(*args, **kwargs):
            raise fulltext.requests.RequestException("no route")

        monkeypatch.setattr(fulltext.requests, "get", boom)
        assert fulltext.fetch(_paper()) is None

    def test_a_real_rendering_comes_back_packed(self, monkeypatch):
        class Response:
            status_code = 200
            headers = {"Content-Type": "text/html; charset=utf-8"}
            text = HTML.replace(
                "Training runs for 3 epochs on 12k examples.",
                "Training runs for 3 epochs on 12k examples. " + "Detail. " * 400,
            )

        monkeypatch.setattr(fulltext.requests, "get", lambda *a, **k: Response())
        body = fulltext.fetch(_paper())
        assert body is not None
        assert "LoRA rank 16" in body


def _paper():
    from datetime import datetime, timezone

    from arxiv_digest.arxiv import Paper

    return Paper(
        arxiv_id="2508.00001",
        version="v1",
        title="A Small Model That Cites Its Sources",
        authors=("Ada Rivers",),
        abstract="We show that requiring a quote removes fabrication.",
        categories=("cs.CL",),
        primary_category="cs.CL",
        published=datetime(2026, 8, 14, tzinfo=timezone.utc),
        abs_url="https://arxiv.org/abs/2508.00001v1",
        pdf_url="https://arxiv.org/pdf/2508.00001v1",
    )

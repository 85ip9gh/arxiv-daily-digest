from datetime import datetime, timezone

import pytest

from arxiv_digest import agent
from arxiv_digest.arxiv import Paper
from arxiv_digest.contrary import Article
from arxiv_digest.hackernews import Story
from arxiv_digest.llm import LLMConfig, LLMError

CONFIG = LLMConfig(backend="ollama", model="qwen3:8b")

ABSTRACT = (
    "We show that requiring a model to quote the source sentence removes most "
    "fabricated fields. On a set of 400 documents the error rate falls from 18 "
    "percent to under 2 percent."
)
BODY = (
    "## Method\nWe fine-tune a 7B decoder with LoRA rank 16 on 12k examples.\n\n"
    "## Results\nAccuracy reaches 91.4 against a 84.2 baseline on HotpotQA."
)


def paper(n: int = 0) -> Paper:
    return Paper(
        arxiv_id=f"2508.0000{n}",
        version="v1",
        title=f"Paper {n}",
        authors=("Ada Rivers",),
        abstract=ABSTRACT,
        categories=("cs.CL",),
        primary_category="cs.CL",
        published=datetime(2026, 8, 14, tzinfo=timezone.utc),
        abs_url="https://arxiv.org/abs/2508.00000",
        pdf_url="https://arxiv.org/pdf/2508.00000",
    )


def story(n: int = 0) -> Story:
    return Story(
        hn_id=f"90000{n}",
        title=f"Story {n}",
        url=f"https://example.com/{n}",
        points=100 + n,
        num_comments=10 + n,
        author="pg",
        created=datetime(2026, 8, 14, tzinfo=timezone.utc),
    )


def deepdive(n: int = 0) -> Article:
    return Article(
        article_id=f"deep-dive-{n}",
        title=f"Deep Dive {n}",
        url=f"https://research.contrary.com/report/deep-dive-{n}",
        published=datetime(2026, 8, 14, tzinfo=timezone.utc),
        authors=("Ada Rivers",),
    )


def fields(
    quote: str,
    *,
    result: str = "Errors fall to under 2 percent.",
    numbers=None,
    method=None,
) -> dict:
    return {
        "problem": "Small models invent fields.",
        "approach": "Require a quote and check it against the source.",
        "method_details": ["a quote is required per field", "measured across 400 documents"]
        if method is None
        else method,
        "result": result,
        "numbers": ["error rate 18 percent to 2 percent"] if numbers is None else numbers,
        "limitations": "Only tested on extraction, not classification.",
        "so_what": "Extraction pipelines get a correct value or none.",
        "quote": quote,
    }


def stub(responses, calls=None):
    """Replace `complete` with a queue of canned responses."""
    queue = list(responses)

    def fake(prompt, schema, *, config, system=None):
        if calls is not None:
            calls.append(prompt)
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    return fake


def test_default_interests_favour_a_working_engineer_over_ml_theory():
    text = agent.DEFAULT_INTERESTS.lower()
    for term in ("software engineering", "devops", "agent tooling", "verification"):
        assert term in text
    assert "benchmark chasing" in text


class TestQuoteGrounding:
    def test_verbatim_quote_passes(self):
        assert agent.quote_is_grounded("the error rate falls from 18 percent", ABSTRACT)

    def test_punctuation_and_case_differences_still_pass(self):
        assert agent.quote_is_grounded(
            "Requiring a model, to quote the source sentence!", ABSTRACT
        )

    def test_paraphrase_fails(self):
        assert not agent.quote_is_grounded(
            "the error rate dropped by roughly nine times", ABSTRACT
        )

    def test_short_quote_is_rejected_even_when_present(self):
        assert not agent.quote_is_grounded("We show that", ABSTRACT)

    def test_a_five_word_fragment_is_not_evidence(self):
        # Present in the abstract, and still too thin to earn the chip.
        assert "requiring a model to quote" in ABSTRACT
        assert not agent.quote_is_grounded("requiring a model to quote", ABSTRACT)


class TestNumberGrounding:
    def test_figures_present_in_the_source_pass(self):
        assert agent.ungrounded_numbers("18 percent to 2 percent on 400 docs", ABSTRACT) == []

    def test_an_invented_benchmark_score_is_caught(self):
        assert agent.ungrounded_numbers("scores 91.7 on MMLU", ABSTRACT) == ["91.7"]

    def test_trailing_zeros_are_the_same_claim(self):
        assert agent.ungrounded_numbers("the rate is 18.0 percent", ABSTRACT) == []

    def test_a_suffixed_size_matches_the_spelled_out_source(self):
        assert agent.ungrounded_numbers("a 7B model", "we train a 7 B parameter model") == []

    def test_every_stray_figure_is_reported_once_and_sorted(self):
        stray = agent.ungrounded_numbers("77.7 and 88.8 and 77.7", ABSTRACT)
        assert stray == ["77.7", "88.8"]


class TestDashes:
    """The house rule against long dashes overrides even a verbatim quote."""

    EN = chr(0x2013)
    EM = chr(0x2014)

    def test_a_numeric_range_becomes_a_word(self):
        assert agent.clean_dashes(f"55.4{self.EN}73.2 points") == "55.4 to 73.2 points"

    def test_an_aside_becomes_a_comma(self):
        assert agent.clean_dashes(f"the model {self.EM} ours {self.EM} wins") == (
            "the model, ours, wins"
        )

    def test_a_compound_name_keeps_a_hyphen(self):
        assert agent.clean_dashes(f"Newton{self.EN}Raphson") == "Newton-Raphson"

    def test_text_without_them_is_untouched(self):
        assert agent.clean_dashes("plain text, 55.4 to 73.2") == "plain text, 55.4 to 73.2"

    def test_a_quote_is_cleaned_on_the_way_out(self, monkeypatch):
        payload = fields(f"the error rate falls from 18{self.EN}percent to under 2 percent")
        payload["result"] = f"Errors fall 18{self.EN}2 percent."
        monkeypatch.setattr(agent, "complete", stub([payload]))
        summary = agent.summarize(paper(), config=CONFIG, read_body=False)
        for text in (summary.quote, summary.result):
            assert self.EN not in text
            assert self.EM not in text

    def test_cleaning_does_not_break_the_checks(self, monkeypatch):
        # The quote arrives with a dash the abstract does not have, and still
        # verifies, because both checks compare on words rather than punctuation.
        payload = fields(f"the error rate falls{self.EM}from 18 percent")
        monkeypatch.setattr(agent, "complete", stub([payload]))
        assert agent.summarize(paper(), config=CONFIG, read_body=False).grounded


class TestSelect:
    def test_returns_the_chosen_papers_with_reasons(self, monkeypatch):
        monkeypatch.setattr(
            agent,
            "complete",
            stub([{"picks": [{"index": 2, "reason": "has numbers"}, {"index": 0, "reason": "new benchmark"}]}]),
        )
        picks = agent.select([paper(i) for i in range(5)], config=CONFIG, count=2)
        assert [p.arxiv_id for p, _ in picks] == ["2508.00002", "2508.00000"]
        assert picks[0][1] == "has numbers"

    def test_out_of_range_and_duplicate_indices_trigger_a_retry(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            agent,
            "complete",
            stub(
                [
                    {"picks": [{"index": 99, "reason": "x"}, {"index": 1, "reason": "y"}, {"index": 1, "reason": "y"}]},
                    {"picks": [{"index": 3, "reason": "ok"}, {"index": 4, "reason": "ok"}]},
                ],
                calls,
            ),
        )
        picks = agent.select([paper(i) for i in range(5)], config=CONFIG, count=2)
        assert [p.arxiv_id for p, _ in picks] == ["2508.00003", "2508.00004"]
        assert "rejected" in calls[1]

    def test_falls_back_to_recency_when_the_model_keeps_failing(self, monkeypatch):
        monkeypatch.setattr(agent, "complete", stub([LLMError("daemon down")]))
        picks = agent.select([paper(i) for i in range(5)], config=CONFIG, count=2)
        assert [p.arxiv_id for p, _ in picks] == ["2508.00000", "2508.00001"]
        assert "recency" in picks[0][1]

    def test_a_thin_day_skips_the_model_entirely(self, monkeypatch):
        monkeypatch.setattr(agent, "complete", stub([]))
        picks = agent.select([paper(0), paper(1)], config=CONFIG, count=3)
        assert len(picks) == 2


class TestSelectStories:
    """select_stories mirrors select()'s shape: same shortlist, retry and
    fall back to recency, no summarize step on this side."""

    def test_returns_the_chosen_stories_with_reasons(self, monkeypatch):
        monkeypatch.setattr(
            agent,
            "complete",
            stub([{"picks": [{"index": 2, "reason": "worth a click"}, {"index": 0, "reason": "relevant"}]}]),
        )
        picks = agent.select_stories([story(i) for i in range(10)], config=CONFIG, count=2)
        assert [s.hn_id for s, _ in picks] == ["900002", "900000"]
        assert picks[0][1] == "worth a click"

    def test_out_of_range_and_duplicate_indices_trigger_a_retry(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            agent,
            "complete",
            stub(
                [
                    {"picks": [{"index": 99, "reason": "x"}, {"index": 1, "reason": "y"}, {"index": 1, "reason": "y"}]},
                    {"picks": [{"index": 3, "reason": "ok"}, {"index": 4, "reason": "ok"}]},
                ],
                calls,
            ),
        )
        picks = agent.select_stories([story(i) for i in range(10)], config=CONFIG, count=2)
        assert [s.hn_id for s, _ in picks] == ["900003", "900004"]
        assert "rejected" in calls[1]

    def test_falls_back_to_recency_when_the_model_keeps_failing(self, monkeypatch):
        monkeypatch.setattr(agent, "complete", stub([LLMError("daemon down")]))
        picks = agent.select_stories([story(i) for i in range(10)], config=CONFIG, count=2)
        assert [s.hn_id for s, _ in picks] == ["900000", "900001"]
        assert "recency" in picks[0][1]

    def test_a_thin_day_skips_the_model_entirely(self, monkeypatch):
        monkeypatch.setattr(agent, "complete", stub([]))
        picks = agent.select_stories([story(0), story(1)], config=CONFIG, count=3)
        assert len(picks) == 2

    def test_reason_is_cleaned_of_long_dashes(self, monkeypatch):
        em = chr(0x2014)
        monkeypatch.setattr(
            agent,
            "complete",
            stub([{"picks": [{"index": 0, "reason": f"good {em} worth a look"}, {"index": 1, "reason": "fine"}]}]),
        )
        picks = agent.select_stories([story(i) for i in range(10)], config=CONFIG, count=2)
        assert em not in picks[0][1]
        assert picks[0][1] == "good, worth a look"


class TestSelectArticles:
    """select_articles mirrors select_stories: same shortlist, retry and fall
    back to recency, no summarize step, picking Contrary Research deep dives."""

    def test_returns_the_chosen_articles_with_reasons(self, monkeypatch):
        monkeypatch.setattr(
            agent,
            "complete",
            stub([{"picks": [{"index": 2, "reason": "sharp on AI infra"}, {"index": 0, "reason": "relevant"}]}]),
        )
        picks = agent.select_articles([deepdive(i) for i in range(10)], config=CONFIG, count=2)
        assert [a.article_id for a, _ in picks] == ["deep-dive-2", "deep-dive-0"]
        assert picks[0][1] == "sharp on AI infra"

    def test_out_of_range_and_duplicate_indices_trigger_a_retry(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            agent,
            "complete",
            stub(
                [
                    {"picks": [{"index": 99, "reason": "x"}, {"index": 1, "reason": "y"}, {"index": 1, "reason": "y"}]},
                    {"picks": [{"index": 3, "reason": "ok"}, {"index": 4, "reason": "ok"}]},
                ],
                calls,
            ),
        )
        picks = agent.select_articles([deepdive(i) for i in range(10)], config=CONFIG, count=2)
        assert [a.article_id for a, _ in picks] == ["deep-dive-3", "deep-dive-4"]
        assert "rejected" in calls[1]

    def test_falls_back_to_recency_when_the_model_keeps_failing(self, monkeypatch):
        monkeypatch.setattr(agent, "complete", stub([LLMError("daemon down")]))
        picks = agent.select_articles([deepdive(i) for i in range(10)], config=CONFIG, count=2)
        assert [a.article_id for a, _ in picks] == ["deep-dive-0", "deep-dive-1"]
        assert "recency" in picks[0][1]

    def test_a_thin_day_skips_the_model_entirely(self, monkeypatch):
        monkeypatch.setattr(agent, "complete", stub([]))
        picks = agent.select_articles([deepdive(0), deepdive(1)], config=CONFIG, count=3)
        assert len(picks) == 2

    def test_reason_is_cleaned_of_long_dashes(self, monkeypatch):
        em = chr(0x2014)
        monkeypatch.setattr(
            agent,
            "complete",
            stub([{"picks": [{"index": 0, "reason": f"deep {em} but readable"}, {"index": 1, "reason": "fine"}]}]),
        )
        picks = agent.select_articles([deepdive(i) for i in range(10)], config=CONFIG, count=2)
        assert em not in picks[0][1]
        assert picks[0][1] == "deep, but readable"

    def test_default_interests_lean_to_tech_and_ai(self):
        text = agent.DEFAULT_CONTRARY_INTERESTS.lower()
        assert "artificial intelligence" in text
        assert "software" in text


class TestSummarize:
    def test_grounded_summary_keeps_everything(self, monkeypatch):
        monkeypatch.setattr(
            agent,
            "complete",
            stub([fields("the error rate falls from 18 percent to under 2 percent")]),
        )
        summary = agent.summarize(paper(), config=CONFIG, read_body=False)
        assert summary.grounded
        assert summary.unverified_numbers == ()
        assert summary.method_details == (
            "a quote is required per field",
            "measured across 400 documents",
        )
        assert summary.limitations.startswith("Only tested")
        assert summary.source_label == "abstract only"

    def test_the_body_is_part_of_the_source_the_checks_run_against(self, monkeypatch):
        monkeypatch.setattr(
            agent,
            "complete",
            stub(
                [
                    fields(
                        "Accuracy reaches 91.4 against a 84.2 baseline",
                        result="Accuracy reaches 91.4 against 84.2.",
                        numbers=["HotpotQA 91.4 vs 84.2"],
                        method=["LoRA rank 16", "12k examples"],
                    )
                ]
            ),
        )
        summary = agent.summarize(paper(), config=CONFIG, body=BODY)
        assert summary.grounded
        assert summary.unverified_numbers == ()
        assert summary.read_full_text
        assert summary.source_label == "full text"

    def test_invented_quote_is_retried_then_dropped(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            agent,
            "complete",
            stub([fields("we evaluated on ImageNet at scale"), fields("also invented text here")], calls),
        )
        summary = agent.summarize(paper(), config=CONFIG, read_body=False)
        assert not summary.grounded
        assert summary.quote == ""
        # The prose is still returned, only the citation is discarded.
        assert summary.result == "Errors fall to under 2 percent."
        assert "does not appear in the source" in calls[1]

    def test_an_invented_figure_is_retried_then_flagged(self, monkeypatch):
        calls = []
        good_quote = "the error rate falls from 18 percent to under 2 percent"
        monkeypatch.setattr(
            agent,
            "complete",
            stub(
                [
                    fields(good_quote, result="It reaches 91.7 on MMLU."),
                    fields(good_quote, result="It reaches 91.7 on MMLU."),
                ],
                calls,
            ),
        )
        summary = agent.summarize(paper(), config=CONFIG, read_body=False)
        assert summary.grounded
        assert summary.unverified_numbers == ("91.7",)
        assert "these figures are not in the source: 91.7" in calls[1]

    def test_second_attempt_can_recover(self, monkeypatch):
        monkeypatch.setattr(
            agent,
            "complete",
            stub([fields("invented"), fields("requiring a model to quote the source sentence")]),
        )
        assert agent.summarize(paper(), config=CONFIG, read_body=False).grounded

    def test_empty_field_is_retried(self, monkeypatch):
        bad = fields("the error rate falls from 18 percent")
        bad["limitations"] = "  "
        calls = []
        monkeypatch.setattr(
            agent,
            "complete",
            stub([bad, fields("the error rate falls from 18 percent")], calls),
        )
        assert agent.summarize(paper(), config=CONFIG, read_body=False).grounded
        assert "empty" in calls[1]

    def test_a_string_where_a_list_belongs_is_tolerated(self, monkeypatch):
        payload = fields("the error rate falls from 18 percent", numbers="18 percent")
        payload["method_details"] = "one quote per field"
        monkeypatch.setattr(agent, "complete", stub([payload]))
        summary = agent.summarize(paper(), config=CONFIG, read_body=False)
        assert summary.method_details == ("one quote per field",)
        assert summary.numbers == ("18 percent",)

    def test_transport_failure_raises(self, monkeypatch):
        monkeypatch.setattr(agent, "complete", stub([LLMError("timeout")]))
        with pytest.raises(LLMError):
            agent.summarize(paper(), config=CONFIG, read_body=False)

    def test_no_network_is_touched_when_the_body_is_skipped(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("fulltext.fetch must not be called")

        monkeypatch.setattr(agent.fulltext, "fetch", explode)
        monkeypatch.setattr(
            agent,
            "complete",
            stub([fields("the error rate falls from 18 percent to under 2 percent")]),
        )
        assert agent.summarize(paper(), config=CONFIG, read_body=False).grounded


class TestBudgetExhaustionPropagates:
    """A dead allowance must not be flattened into "this paper failed"."""

    def test_rate_limit_exhausted_escapes_summarize(self, monkeypatch):
        from arxiv_digest.llm import RateLimitExhausted

        def broke(*a, **kw):
            raise RateLimitExhausted("tokens per day (TPD): Limit 100000")

        monkeypatch.setattr(agent, "complete", broke)
        with pytest.raises(RateLimitExhausted):
            agent.summarize(paper(), config=CONFIG, body="some source text")

    def test_an_ordinary_model_error_still_becomes_a_paper_failure(self, monkeypatch):
        from arxiv_digest.llm import LLMError, RateLimitExhausted

        def broke(*a, **kw):
            raise LLMError("returned non-JSON")

        monkeypatch.setattr(agent, "complete", broke)
        with pytest.raises(LLMError) as caught:
            agent.summarize(paper(), config=CONFIG, body="some source text")
        assert not isinstance(caught.value, RateLimitExhausted)

    def test_the_reason_survives_into_the_message(self, monkeypatch):
        """A retired model fails on every paper, so the why has to reach the log."""

        def broke(*a, **kw):
            raise LLMError("call failed: HTTP 404 body='model_not_found'")

        monkeypatch.setattr(agent, "complete", broke)
        with pytest.raises(LLMError) as caught:
            agent.summarize(paper(), config=CONFIG, body="some source text")
        assert "model_not_found" in str(caught.value)

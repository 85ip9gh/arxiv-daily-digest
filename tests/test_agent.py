from datetime import datetime, timezone

import pytest

from arxiv_digest import agent
from arxiv_digest.arxiv import Paper
from arxiv_digest.llm import LLMConfig, LLMError

CONFIG = LLMConfig()

ABSTRACT = (
    "We show that requiring a model to quote the source sentence removes most "
    "fabricated fields. On a set of 400 documents the error rate falls from 18 "
    "percent to under 2 percent."
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


def fields(quote: str) -> dict:
    return {
        "problem": "Small models invent fields.",
        "approach": "Require a quote and check it.",
        "result": "Errors fall to under 2 percent.",
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
        monkeypatch.setattr(
            agent, "complete", stub([LLMError("daemon down")])
        )
        picks = agent.select([paper(i) for i in range(5)], config=CONFIG, count=2)
        assert [p.arxiv_id for p, _ in picks] == ["2508.00000", "2508.00001"]
        assert "recency" in picks[0][1]

    def test_a_thin_day_skips_the_model_entirely(self, monkeypatch):
        monkeypatch.setattr(agent, "complete", stub([]))
        picks = agent.select([paper(0), paper(1)], config=CONFIG, count=3)
        assert len(picks) == 2


class TestSummarize:
    def test_grounded_quote_is_kept(self, monkeypatch):
        monkeypatch.setattr(
            agent,
            "complete",
            stub([fields("the error rate falls from 18 percent to under 2 percent")]),
        )
        summary = agent.summarize(paper(), config=CONFIG)
        assert summary.grounded
        assert summary.quote.startswith("the error rate falls")

    def test_invented_quote_is_retried_then_dropped(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            agent,
            "complete",
            stub([fields("we evaluated on ImageNet at scale"), fields("also invented text here")], calls),
        )
        summary = agent.summarize(paper(), config=CONFIG)
        assert not summary.grounded
        assert summary.quote == ""
        # The prose is still returned, only the citation is discarded.
        assert summary.result == "Errors fall to under 2 percent."
        assert "does not appear in the abstract" in calls[1]

    def test_second_attempt_can_recover(self, monkeypatch):
        monkeypatch.setattr(
            agent,
            "complete",
            stub([fields("invented"), fields("requiring a model to quote the source sentence")]),
        )
        assert agent.summarize(paper(), config=CONFIG).grounded

    def test_empty_field_is_retried(self, monkeypatch):
        bad = fields("the error rate falls from 18 percent")
        bad["result"] = "  "
        calls = []
        monkeypatch.setattr(
            agent,
            "complete",
            stub([bad, fields("the error rate falls from 18 percent")], calls),
        )
        assert agent.summarize(paper(), config=CONFIG).grounded
        assert "empty" in calls[1]

    def test_transport_failure_raises(self, monkeypatch):
        monkeypatch.setattr(agent, "complete", stub([LLMError("timeout")]))
        with pytest.raises(LLMError):
            agent.summarize(paper(), config=CONFIG)

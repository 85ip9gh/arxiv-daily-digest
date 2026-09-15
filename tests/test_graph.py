"""The two graphs: the per-paper summarizer loop and the day pipeline.

The summarizer cases overlap with test_agent on purpose: agent.summarize now
delegates to graph.run_summary, so these prove the graph is the real path and
not a wrapper the tests route around. The pipeline cases inject fake source
runners so the routing (order, and the publish-or-nothing conditional edge) is
tested without a model or a network.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from arxiv_digest import agent, graph
from arxiv_digest.arxiv import Paper
from arxiv_digest.llm import LLMConfig, LLMError, RateLimitExhausted

CONFIG = LLMConfig(backend="ollama", model="qwen3:8b")

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


def fields(quote: str, *, result: str = "Errors fall to under 2 percent.") -> dict:
    return {
        "problem": "Small models invent fields.",
        "approach": "Require a quote and check it against the source.",
        "method_details": ["a quote is required per field"],
        "result": result,
        "numbers": ["error rate 18 percent to 2 percent"],
        "limitations": "Only tested on extraction.",
        "so_what": "Extraction pipelines get a correct value or none.",
        "quote": quote,
    }


def stub(responses, calls=None):
    queue = list(responses)

    def fake(prompt, schema, *, config, system=None):
        if calls is not None:
            calls.append(prompt)
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    return fake


class TestSummarizerGraph:
    def test_a_grounded_paper_is_accepted(self, monkeypatch):
        monkeypatch.setattr(
            agent, "complete",
            stub([fields("the error rate falls from 18 percent to under 2 percent")]),
        )
        summary = graph.run_summary(paper(), config=CONFIG, read_body=False)
        assert summary.grounded
        assert summary.quote

    def test_a_bad_citation_retries_then_gives_up_but_keeps_the_prose(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            agent, "complete",
            stub([fields("invented one"), fields("invented two")], calls),
        )
        summary = graph.run_summary(paper(), config=CONFIG, read_body=False)
        assert not summary.grounded
        assert summary.quote == ""
        assert summary.result == "Errors fall to under 2 percent."
        assert len(calls) == 2
        assert "does not appear in the source" in calls[1]

    def test_a_recovered_retry_is_grounded(self, monkeypatch):
        monkeypatch.setattr(
            agent, "complete",
            stub([fields("invented"), fields("requiring a model to quote the source sentence")]),
        )
        assert graph.run_summary(paper(), config=CONFIG, read_body=False).grounded

    def test_an_ordinary_error_raises_llm_error(self, monkeypatch):
        monkeypatch.setattr(agent, "complete", stub([LLMError("returned non-JSON")]))
        with pytest.raises(LLMError) as caught:
            graph.run_summary(paper(), config=CONFIG, read_body=False)
        assert not isinstance(caught.value, RateLimitExhausted)
        assert "could not summarize" in str(caught.value)

    def test_a_dead_allowance_propagates_as_rate_limit_exhausted(self, monkeypatch):
        monkeypatch.setattr(
            agent, "complete",
            stub([RateLimitExhausted("tokens per day (TPD): Limit 100000")]),
        )
        with pytest.raises(RateLimitExhausted):
            graph.run_summary(paper(), config=CONFIG, read_body=False)


class TestPipelineGraph:
    def _state(self, **over):
        base = {
            "args": SimpleNamespace(out_dir="digests"),
            "config": CONFIG,
            "seen": set(),
            "hn_seen": set(),
            "contrary_seen": set(),
            "summaries": [],
            "hn_picks": [],
            "contrary_picks": [],
            "exit_code": 0,
        }
        base.update(over)
        return base

    def test_sources_run_in_order_then_publish_when_there_is_content(self):
        order = []
        pipe = graph.build_pipeline(
            run_arxiv=lambda a, c, s: order.append("arxiv") or ["summary"],
            run_hn=lambda a, c, s: order.append("hn") or [],
            run_contrary=lambda a, c, s: order.append("contrary") or [],
            publish=lambda state: order.append("publish") or 0,
        )
        final = pipe.invoke(self._state())
        assert order == ["arxiv", "hn", "contrary", "publish"]
        assert final["exit_code"] == 0

    def test_all_empty_skips_publish_and_exits_nonzero(self):
        published = []
        pipe = graph.build_pipeline(
            run_arxiv=lambda a, c, s: [],
            run_hn=lambda a, c, s: [],
            run_contrary=lambda a, c, s: [],
            publish=lambda state: published.append(True) or 0,
        )
        final = pipe.invoke(self._state())
        assert final["exit_code"] == 1
        assert published == []

    def test_a_single_late_source_still_publishes(self):
        pipe = graph.build_pipeline(
            run_arxiv=lambda a, c, s: [],
            run_hn=lambda a, c, s: [],
            run_contrary=lambda a, c, s: ["a deep dive"],
            publish=lambda state: 0,
        )
        assert pipe.invoke(self._state())["exit_code"] == 0

    def test_publish_receives_the_source_output_on_the_state(self):
        seen = {}
        pipe = graph.build_pipeline(
            run_arxiv=lambda a, c, s: ["paper-summary"],
            run_hn=lambda a, c, s: [("story", "why")],
            run_contrary=lambda a, c, s: [],
            publish=lambda state: seen.update(state) or 0,
        )
        pipe.invoke(self._state())
        assert seen["summaries"] == ["paper-summary"]
        assert seen["hn_picks"] == [("story", "why")]

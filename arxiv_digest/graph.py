"""The pipeline and the per-paper loop, expressed as LangGraph state graphs.

The control flow of this project was always a graph. The day is three
independent sources converging on one archive, and each paper is a
generate-then-check loop that retries once on a failed citation. Both were
written as plain Python first, which is the honest order: the graph earns its
place only once the flow it draws is the flow that already runs.

Two graphs live here.

**The pipeline.** arXiv, Hacker News and Contrary Research each run as a node,
in sequence, because the token pacing in `llm.py` keeps a single process-wide
window and is not built for concurrent callers. A conditional edge then routes
to `publish` when any source produced something and to `nothing` when all three
came back empty, which is the one case that exits nonzero.

**The summarizer.** One paper's `generate -> verify -> retry` loop, the
reflection pattern with a real conditional edge: verify sends the run back to
generate with the failure quoted when a citation or a figure does not check out,
and forward to accept once it does or the attempts run out.

What this deliberately does NOT do is move model calls or their pacing into the
framework. `llm.complete` and its `TokenWindow` stay exactly where they are, and
every node reaches them through the same `agent` functions the tests already
pin. The graph owns the control flow; the tuned, measured pieces own themselves.
The nodes take `agent` by a late import rather than a module-level one, so this
module never imports `agent` while `agent` is importing it.
"""
from __future__ import annotations

import sys
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import LLMError, RateLimitExhausted


# ---------------------------------------------------------------------------
# The per-paper summarizer: generate, verify, retry once, accept.
# ---------------------------------------------------------------------------
class SummarizeState(TypedDict):
    """One paper moving through the loop.

    `paper`, `source`, `prompt`, `read_full_text`, `reason` and `config` are the
    fixed inputs; the rest is what the loop updates. `decision` is how a node
    tells the router, and how `run_summary` learns whether the run ended in a
    summary, a dead allowance, or an ordinary model error.
    """

    paper: Any
    source: str
    prompt: str
    read_full_text: bool
    reason: str
    config: Any
    attempts_left: int
    last_error: str
    fields: Any
    failure: str
    decision: str
    result: Any


def _sum_generate(state: SummarizeState) -> dict:
    """One model call. A dead allowance and an ordinary error are recorded on the
    state rather than raised, so the exact exception is raised once, by
    `run_summary`, instead of from inside the graph where it could be wrapped."""
    from . import agent

    prompt = state["prompt"]
    if state["last_error"]:
        prompt = f"{prompt}\n\nYour last answer was rejected: {state['last_error']}"

    try:
        fields = agent.complete(
            prompt, agent.SUMMARY_SCHEMA, config=state["config"], system=agent.SYSTEM
        )
    except RateLimitExhausted as exc:
        return {"fields": None, "failure": str(exc), "decision": "ratelimit",
                "attempts_left": state["attempts_left"] - 1}
    except LLMError as exc:
        return {"fields": None, "failure": str(exc), "decision": "error",
                "attempts_left": state["attempts_left"] - 1}
    return {"fields": fields, "attempts_left": state["attempts_left"] - 1}


def _after_generate(state: SummarizeState) -> str:
    return "stop" if state["fields"] is None else "verify"


def _sum_verify(state: SummarizeState) -> dict:
    """Run the two checks and decide: accept, or retry with the failure quoted.

    Empty required fields are their own retry and skip the citation and figure
    checks, matching the original loop. A retry is only offered while an attempt
    remains; on the last pass the summary is accepted and whatever failed is
    marked on the page by `accept` rather than retried into oblivion.
    """
    from . import agent

    fields = state["fields"]
    missing = [k for k in agent.REQUIRED_TEXT if not str(fields.get(k, "")).strip()]
    if missing:
        if state["attempts_left"] > 0:
            return {"last_error": f"these fields were empty: {', '.join(missing)}",
                    "decision": "retry"}
        return {"decision": "accept"}

    quote = str(fields.get("quote", "")).strip().strip('"')
    quote_ok = agent.quote_is_grounded(quote, state["source"])
    stray = agent.ungrounded_numbers(agent._checked_text(fields), state["source"])
    if quote_ok and not stray:
        return {"decision": "accept"}
    if state["attempts_left"] > 0:
        problems = []
        if not quote_ok:
            problems.append(
                "your quote does not appear in the source word for word. Copy a "
                "fragment straight out of the text"
            )
        if stray:
            problems.append(
                "these figures are not in the source: "
                f"{', '.join(stray)}. Use only values the text gives"
            )
        return {"last_error": ". ".join(problems), "decision": "retry"}
    return {"decision": "accept"}


def _after_verify(state: SummarizeState) -> str:
    return "generate" if state["decision"] == "retry" else "accept"


def _sum_accept(state: SummarizeState) -> dict:
    """Build the summary from the final fields, marking what could not be verified.

    The checks run once more here so the success path and the give-up path build
    the same way: a grounded quote is kept, an ungrounded one is dropped to the
    empty string, and stray figures are recorded rather than presented as real.
    """
    from . import agent

    fields = state["fields"]
    quote = str(fields.get("quote", "")).strip().strip('"')
    quote_ok = agent.quote_is_grounded(quote, state["source"])
    stray = agent.ungrounded_numbers(agent._checked_text(fields), state["source"])
    summary = agent._build(
        state["paper"], fields, quote if quote_ok else "", state["reason"],
        quote_ok, tuple(stray), state["read_full_text"],
    )
    return {"result": summary, "decision": "done"}


def _build_summarizer():
    graph = StateGraph(SummarizeState)
    graph.add_node("generate", _sum_generate)
    graph.add_node("verify", _sum_verify)
    graph.add_node("accept", _sum_accept)
    graph.add_edge(START, "generate")
    graph.add_conditional_edges("generate", _after_generate, {"stop": END, "verify": "verify"})
    graph.add_conditional_edges("verify", _after_verify, {"generate": "generate", "accept": "accept"})
    graph.add_edge("accept", END)
    return graph.compile()


_SUMMARIZER = _build_summarizer()


def run_summary(paper, *, config, reason: str = "", attempts: int = 2,
                body: str | None = None, read_body: bool = True):
    """Summarize one paper through the graph, verifying quote and figures.

    Same contract as the original `agent.summarize`: `body` is injected by the
    tests, production reads it from arXiv's HTML rendering, a paper without one
    falls back to its abstract. A dead allowance raises `RateLimitExhausted`, an
    ordinary model error raises `LLMError`; both are raised here rather than in a
    node so the caller sees the exact type.
    """
    from . import agent

    if body is None and read_body:
        body = agent.fulltext.fetch(paper)
    read_full_text = bool(body)
    source = f"{paper.abstract}\n\n{body}" if body else paper.abstract
    prompt = agent._prompt_for(paper, source, read_full_text)

    final = _SUMMARIZER.invoke({
        "paper": paper, "source": source, "prompt": prompt,
        "read_full_text": read_full_text, "reason": reason, "config": config,
        "attempts_left": attempts, "last_error": "", "fields": None,
        "failure": "", "decision": "", "result": None,
    })

    if final["decision"] == "ratelimit":
        raise RateLimitExhausted(final["failure"])
    if final["decision"] == "error":
        detail = f": {final['failure']}" if final["failure"] else ""
        raise LLMError(
            f"could not summarize {paper.arxiv_id} with {config.label}{detail}"
        )
    return final["result"]


# ---------------------------------------------------------------------------
# The day pipeline: three sources, then publish or leave the archive alone.
# ---------------------------------------------------------------------------
class PipelineState(TypedDict):
    """One day's run. The three source lists are what each node fills, and
    `exit_code` is what the process returns."""

    args: Any
    config: Any
    seen: set
    hn_seen: set
    contrary_seen: set
    summaries: list
    hn_picks: list
    contrary_picks: list
    exit_code: int


def build_pipeline(
    *,
    run_arxiv: Callable,
    run_hn: Callable,
    run_contrary: Callable,
    publish: Callable,
):
    """Wire the day into a graph, with the source runners and the publish step
    injected so the business logic and its tests stay in `cli`.

    The three sources run in sequence, not in parallel: the token window in
    `llm.py` is process-wide and reactive, so overlapping callers would race it.
    The conditional edge is the whole point, the branch that used to be an `if`
    at the end of `main`: publish when there is anything, leave the archive
    untouched and exit nonzero when there is not.
    """

    def arxiv_node(state: PipelineState) -> dict:
        return {"summaries": run_arxiv(state["args"], state["config"], state["seen"])}

    def hn_node(state: PipelineState) -> dict:
        return {"hn_picks": run_hn(state["args"], state["config"], state["hn_seen"])}

    def contrary_node(state: PipelineState) -> dict:
        return {"contrary_picks": run_contrary(state["args"], state["config"], state["contrary_seen"])}

    def gate(state: PipelineState) -> str:
        if state["summaries"] or state["hn_picks"] or state["contrary_picks"]:
            return "publish"
        return "nothing"

    def publish_node(state: PipelineState) -> dict:
        return {"exit_code": publish(state)}

    def nothing_node(state: PipelineState) -> dict:
        print(
            f"nothing to publish today, leaving {state['args'].out_dir} untouched",
            file=sys.stderr,
        )
        return {"exit_code": 1}

    graph = StateGraph(PipelineState)
    graph.add_node("arxiv", arxiv_node)
    graph.add_node("hackernews", hn_node)
    graph.add_node("contrary", contrary_node)
    graph.add_node("publish", publish_node)
    graph.add_node("nothing", nothing_node)
    graph.add_edge(START, "arxiv")
    graph.add_edge("arxiv", "hackernews")
    graph.add_edge("hackernews", "contrary")
    graph.add_conditional_edges("contrary", gate, {"publish": "publish", "nothing": "nothing"})
    graph.add_edge("publish", END)
    graph.add_edge("nothing", END)
    return graph.compile()

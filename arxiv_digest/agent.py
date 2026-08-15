"""The two model steps, and the checks that make an 8B model's output usable.

The agent is a short loop, not a chat: select three papers, summarize each,
verify the summary against the abstract, retry once with the failure quoted
back. Everything the model returns is either checked or discarded.

The load-bearing check is the citation. Each summary has to quote the sentence
fragment its main claim came from, and that fragment has to appear in the
abstract. A small model asked to summarize a paper it half-recognizes will
happily describe the paper it remembers instead of the one in front of it, and
that failure is invisible in fluent prose. It is not invisible in a string
comparison.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .arxiv import Paper
from .llm import LLMConfig, LLMError, complete

DEFAULT_INTERESTS = (
    "LLM agents and tool use, retrieval, evaluation and benchmarks, "
    "efficient inference on small or local models, and results an engineer "
    "could apply rather than pure theory"
)

SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "reason"],
            },
        }
    },
    "required": ["picks"],
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "problem": {"type": "string"},
        "approach": {"type": "string"},
        "result": {"type": "string"},
        "so_what": {"type": "string"},
        "quote": {"type": "string"},
    },
    "required": ["problem", "approach", "result", "so_what", "quote"],
}

SYSTEM = (
    "You read machine learning abstracts and report what they say. "
    "You never add findings that are not in the text in front of you. "
    "Never use em-dashes."
)

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class Summary:
    """One summarized paper, with the citation check already applied."""

    paper: Paper
    problem: str
    approach: str
    result: str
    so_what: str
    quote: str  # empty when the model could not produce a grounded one
    reason: str  # why the selector picked it
    grounded: bool


def _normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def quote_is_grounded(quote: str, abstract: str, *, min_words: int = 4) -> bool:
    """True when the quote really appears in the abstract.

    Compared on lowercased alphanumeric words so that a model's punctuation or
    whitespace differences do not fail an otherwise honest citation. Quotes
    shorter than `min_words` are rejected: three common words match almost any
    abstract and prove nothing.
    """
    normal_quote = _normalize(quote)
    if len(normal_quote.split()) < min_words:
        return False
    return normal_quote in _normalize(abstract)


def _candidate_block(papers: list[Paper], abstract_chars: int = 400) -> str:
    lines = []
    for i, paper in enumerate(papers):
        abstract = paper.abstract[:abstract_chars]
        lines.append(
            f"[{i}] {paper.title}\n"
            f"    categories: {', '.join(paper.categories)}\n"
            f"    abstract: {abstract}"
        )
    return "\n\n".join(lines)


def select(
    papers: list[Paper],
    *,
    config: LLMConfig,
    count: int = 3,
    interests: str = DEFAULT_INTERESTS,
    attempts: int = 2,
    shortlist: int = 40,
) -> list[tuple[Paper, str]]:
    """Pick `count` papers and keep the selector's one-line reason for each.

    Only the newest `shortlist` candidates are shown to the model. cs.AI alone
    posts well over a hundred papers a day, and an 8B model handed all of them
    reads none of them properly.

    Falls back to the newest `count` papers if the model cannot return valid
    indices. A digest of the three newest papers is worse than a curated one
    and much better than a crash at 07:00.
    """
    papers = papers[:shortlist]
    if len(papers) <= count:
        return [(p, "only candidate for the day") for p in papers]

    prompt = (
        f"Here are {len(papers)} papers posted to arXiv today.\n\n"
        f"{_candidate_block(papers)}\n\n"
        f"Pick the {count} most worth reading for someone interested in: {interests}.\n"
        f"Prefer concrete results and released artifacts over position papers. "
        f"Do not pick two papers that make the same point.\n"
        f"Give each pick's list index and one sentence saying why it earns a slot."
    )

    last_error = ""
    for attempt in range(attempts):
        try:
            raw = complete(
                prompt if not last_error else f"{prompt}\n\nYour last answer was rejected: {last_error}",
                SELECT_SCHEMA,
                config=config,
                system=SYSTEM,
            )
        except LLMError:
            break

        chosen: list[tuple[Paper, str]] = []
        seen: set[int] = set()
        for pick in raw.get("picks", []):
            try:
                index = int(pick.get("index"))
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(papers) and index not in seen:
                seen.add(index)
                chosen.append((papers[index], str(pick.get("reason", "")).strip()))

        if len(chosen) >= count:
            return chosen[:count]
        last_error = (
            f"you returned {len(chosen)} usable indices, "
            f"needed {count} distinct integers between 0 and {len(papers) - 1}"
        )

    return [(p, "picked by recency, the model's selection was unusable") for p in papers[:count]]


def summarize(
    paper: Paper,
    *,
    config: LLMConfig,
    reason: str = "",
    attempts: int = 2,
) -> Summary:
    """Summarize one paper and verify its quote against the abstract."""
    prompt = (
        f"Title: {paper.title}\n"
        f"Authors: {paper.author_line}\n"
        f"Abstract: {paper.abstract}\n\n"
        "Write four short entries about this paper, each one or two plain sentences:\n"
        "problem: what was not working before this paper.\n"
        "approach: what the authors actually did, specifically.\n"
        "result: what they measured or shipped, with the numbers if the abstract gives any.\n"
        "so_what: who should care and why, in one sentence.\n"
        "quote: copy one fragment of at least eight words from the abstract, word for word, "
        "that supports the result you wrote. Copy it exactly, do not paraphrase it.\n\n"
        "Use only what is in the abstract above. Never use em-dashes."
    )

    last_error = ""
    fields = None
    for attempt in range(attempts):
        try:
            fields = complete(
                prompt if not last_error else f"{prompt}\n\nYour last answer was rejected: {last_error}",
                SUMMARY_SCHEMA,
                config=config,
                system=SYSTEM,
            )
        except LLMError:
            fields = None
            break

        missing = [k for k in ("problem", "approach", "result", "so_what") if not str(fields.get(k, "")).strip()]
        if missing:
            last_error = f"these fields were empty: {', '.join(missing)}"
            continue

        quote = str(fields.get("quote", "")).strip().strip('"')
        if quote_is_grounded(quote, paper.abstract):
            return _build(paper, fields, quote, reason, grounded=True)

        last_error = (
            "your quote does not appear in the abstract word for word. "
            "Copy a fragment straight out of the abstract text."
        )

    if fields is None:
        raise LLMError(f"could not summarize {paper.arxiv_id} with {config.label}")

    # The prose survives, the invented citation does not. A summary with no
    # quote is honestly marked as unverified in the digest.
    return _build(paper, fields, "", reason, grounded=False)


def _build(paper, fields, quote, reason, *, grounded) -> Summary:
    return Summary(
        paper=paper,
        problem=str(fields.get("problem", "")).strip(),
        approach=str(fields.get("approach", "")).strip(),
        result=str(fields.get("result", "")).strip(),
        so_what=str(fields.get("so_what", "")).strip(),
        quote=quote,
        reason=reason,
        grounded=grounded,
    )

"""The two model steps, and the checks that make the output usable.

The agent is a short loop, not a chat: select three papers, read each one's body
where arXiv renders it, summarize, verify the summary against the source, retry
once with the failure quoted back. Everything the model returns is either
checked or marked as unchecked.

Two checks do the work, and the second one is what makes a technical summary
safe to publish:

**The citation.** Each summary quotes the fragment its result came from, and
that fragment has to appear in the source. A model asked to summarize a paper it
half-recognizes will describe the paper it remembers, and that failure is
invisible in fluent prose but obvious in a string comparison.

**The numbers.** Every figure in the result and in the numbers list has to
appear in the source text. Detail is exactly where a summarizer invents: a
plausible benchmark score is the easiest thing in the world to write and the
hardest to notice. Prose can be vague and still be honest. A number cannot.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import fulltext
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
        "method_details": {"type": "array", "items": {"type": "string"}},
        "result": {"type": "string"},
        "numbers": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "string"},
        "so_what": {"type": "string"},
        "quote": {"type": "string"},
    },
    "required": [
        "problem",
        "approach",
        "method_details",
        "result",
        "numbers",
        "limitations",
        "so_what",
        "quote",
    ],
}

SYSTEM = (
    "You are a research engineer reading a machine learning paper for a "
    "colleague who will decide from your notes whether to read it. You are "
    "specific and technical. You name architectures, datasets, baselines and "
    "measured values. You never state a number that is not in the text in "
    "front of you, and you never describe a method the text does not describe. "
    "Never use em-dashes."
)

REQUIRED_TEXT = ("problem", "approach", "result", "so_what", "limitations")

_WORD = re.compile(r"[a-z0-9]+")
# Matches 55.4, 1.5, 128, 12k, 70B. The suffix is captured so that "70B" and
# "70" are not treated as the same claim.
_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:\s?[kKmMbB](?![a-zA-Z]))?")


@dataclass
class Summary:
    """One summarized paper, with both checks already applied."""

    paper: Paper
    problem: str
    approach: str
    result: str
    so_what: str
    limitations: str
    method_details: tuple[str, ...] = ()
    numbers: tuple[str, ...] = ()
    quote: str = ""
    reason: str = ""
    grounded: bool = False
    unverified_numbers: tuple[str, ...] = ()
    read_full_text: bool = False

    @property
    def source_label(self) -> str:
        return "full text" if self.read_full_text else "abstract only"


def _normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def quote_is_grounded(quote: str, source: str, *, min_words: int = 4) -> bool:
    """True when the quote really appears in the source.

    Compared on lowercased alphanumeric words so that a model's punctuation or
    whitespace differences do not fail an otherwise honest citation. Quotes
    shorter than `min_words` are rejected: three common words match almost any
    paper and prove nothing.
    """
    normal_quote = _normalize(quote)
    if len(normal_quote.split()) < min_words:
        return False
    return normal_quote in _normalize(source)


def _number_tokens(text: str) -> list[str]:
    return [_normalize_number(m) for m in _NUMBER.findall(text)]


def _normalize_number(token: str) -> str:
    token = token.replace(" ", "").lower()
    if "." in token:
        head = token.rstrip("kmb")
        suffix = token[len(head):]
        # 82.30 and 82.3 are the same claim written two ways.
        head = head.rstrip("0").rstrip(".") if "." in head else head
        return head + suffix
    return token


def ungrounded_numbers(text: str, source: str) -> list[str]:
    """Figures in `text` that do not appear anywhere in `source`.

    Two passes, because a paper writes the same value several ways. A token
    match catches `55.4` against `55.4`, and a raw substring match catches
    `1.5B` against `1.5 B parameters`.
    """
    known = set(_number_tokens(source))
    raw = source.lower().replace(",", "")
    missing = []
    for found in _NUMBER.findall(text):
        token = _normalize_number(found)
        if token in known:
            continue
        if found.strip().lower().replace(" ", "") in raw.replace(" ", ""):
            continue
        missing.append(found.strip())
    return sorted(set(missing))


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
    posts well over a hundred papers a day, and a model handed all of them
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


def _prompt_for(paper: Paper, body: str, read_full_text: bool) -> str:
    scope = (
        "the body of the paper below" if read_full_text else "the abstract below, "
        "which is all arXiv publishes for this paper in machine-readable form"
    )
    return (
        f"Title: {paper.title}\n"
        f"Authors: {paper.author_line}\n"
        f"Categories: {', '.join(paper.categories)}\n\n"
        f"SOURCE TEXT\n{body}\n\nEND OF SOURCE TEXT\n\n"
        f"Write technical notes on this paper from {scope}.\n\n"
        "problem: what was not working before this paper, and why the obvious "
        "fix does not work. Two or three sentences.\n"
        "approach: how the method actually works, at the level of mechanism. "
        "Name the components and how they fit together. Four to six sentences. "
        "A reader should be able to describe the method after reading this.\n"
        "method_details: three to six short entries carrying the specifics an "
        "engineer would ask for. Model sizes, architectures, datasets, training "
        "or inference setup, baselines compared against, ablations run. One "
        "fact each, no filler entries.\n"
        "result: what was measured and how it came out, with the values. Two to "
        "four sentences.\n"
        "numbers: two to six headline figures, each written as metric, value, "
        "and what it is compared against. Copy the values exactly as the source "
        "gives them.\n"
        "limitations: what the paper does not establish, or says it cannot do. "
        "One or two sentences. If the source states none, say that plainly.\n"
        "so_what: who should care and why, in one or two sentences.\n"
        "quote: copy one fragment of at least eight words from the source, word "
        "for word, that supports the result you wrote. Copy it exactly.\n\n"
        "Use only the source text above. Every number you write must appear in "
        "it. Never use em-dashes."
    )


def summarize(
    paper: Paper,
    *,
    config: LLMConfig,
    reason: str = "",
    attempts: int = 2,
    body: str | None = None,
    read_body: bool = True,
) -> Summary:
    """Summarize one paper and verify its quote and its figures against the source.

    `body` is injected by the tests. In production the body is fetched from
    arXiv's HTML rendering, and papers without one fall back to the abstract.
    """
    if body is None and read_body:
        body = fulltext.fetch(paper)
    read_full_text = bool(body)
    source = f"{paper.abstract}\n\n{body}" if body else paper.abstract

    prompt = _prompt_for(paper, source, read_full_text)
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

        missing = [k for k in REQUIRED_TEXT if not str(fields.get(k, "")).strip()]
        if missing:
            last_error = f"these fields were empty: {', '.join(missing)}"
            continue

        quote = str(fields.get("quote", "")).strip().strip('"')
        quote_ok = quote_is_grounded(quote, source)
        stray = ungrounded_numbers(_checked_text(fields), source)

        if quote_ok and not stray:
            return _build(paper, fields, quote, reason, True, (), read_full_text)

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
        last_error = ". ".join(problems)

    if fields is None:
        raise LLMError(f"could not summarize {paper.arxiv_id} with {config.label}")

    # The prose survives. The citation and the figures that could not be found
    # are marked on the page rather than presented as checked.
    quote = str(fields.get("quote", "")).strip().strip('"')
    quote_ok = quote_is_grounded(quote, source)
    stray = ungrounded_numbers(_checked_text(fields), source)
    return _build(
        paper,
        fields,
        quote if quote_ok else "",
        reason,
        quote_ok,
        tuple(stray),
        read_full_text,
    )


_LONG_DASHES = chr(0x2013) + chr(0x2014)  # en dash, em dash
_RANGE_DASH = re.compile(r"(\d)\s*[" + _LONG_DASHES + r"]\s*(\d)")
_WORD_DASH = re.compile(r"(?<=[A-Za-z])[" + _LONG_DASHES + r"](?=[A-Za-z])")
_LOOSE_DASH = re.compile(r"\s*[" + _LONG_DASHES + r"]\s*")


def clean_dashes(text: str) -> str:
    """Normalize long dashes out of everything the site publishes.

    Papers use en dashes for ranges and em dashes for asides, and both arrive
    inside a verbatim quote. The house rule against them is absolute, so the
    substitution happens here at the boundary rather than in the prompt, where
    a model would comply unreliably. Verification is unaffected: both checks
    compare on words and digits, never on punctuation.
    """
    text = _RANGE_DASH.sub(r"\1 to \2", text)
    text = _WORD_DASH.sub("-", text)
    return _LOOSE_DASH.sub(", ", text)


def _checked_text(fields: dict) -> str:
    """The fields whose figures have to be real.

    Method details are in scope on purpose. A summarizer inventing "LoRA rank
    16" or "12k examples" is the exact failure the extra depth introduces, and
    it is the most convincing thing on the page.
    """
    return " ".join(
        [str(fields.get("result", ""))]
        + _as_list(fields.get("numbers"))
        + _as_list(fields.get("method_details"))
    )


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _text(fields: dict, key: str) -> str:
    return clean_dashes(str(fields.get(key, "")).strip())


def _build(paper, fields, quote, reason, grounded, stray, read_full_text) -> Summary:
    return Summary(
        paper=paper,
        problem=_text(fields, "problem"),
        approach=_text(fields, "approach"),
        result=_text(fields, "result"),
        so_what=_text(fields, "so_what"),
        limitations=_text(fields, "limitations"),
        method_details=tuple(clean_dashes(d) for d in _as_list(fields.get("method_details"))),
        numbers=tuple(clean_dashes(n) for n in _as_list(fields.get("numbers"))),
        quote=clean_dashes(quote),
        reason=clean_dashes(reason),
        grounded=grounded,
        unverified_numbers=tuple(stray),
        read_full_text=read_full_text,
    )

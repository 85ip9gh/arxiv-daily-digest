"""Markdown rendering, and the seen-paper state that stops repeats."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .agent import Summary

SEEN_FILE = "seen.json"
# Two weeks is long enough that a paper cannot come back through a v2 posting
# in the same fortnight, and short enough that the file stays small.
SEEN_LIMIT = 400


def load_seen(out_dir: Path) -> set[str]:
    path = out_dir / SEEN_FILE
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return set()


def save_seen(out_dir: Path, seen: set[str], added: list[str]) -> None:
    """Keep insertion order so the oldest ids fall off the end first."""
    ordered = [i for i in added if i not in seen] + sorted(seen)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SEEN_FILE).write_text(
        json.dumps(ordered[:SEEN_LIMIT], indent=0), encoding="utf-8"
    )


def render(summaries: list[Summary], *, day: date, model_label: str) -> str:
    lines = [
        f"# arXiv AI digest, {day.isoformat()}",
        "",
        f"Three papers, summarized by {model_label}.",
        "",
    ]

    for n, s in enumerate(summaries, start=1):
        p = s.paper
        lines += [
            f"## {n}. {p.title}",
            "",
            f"{p.author_line} | {p.primary_category} | "
            f"[abstract]({p.abs_url}) | [pdf]({p.pdf_url})",
            "",
            f"**Problem.** {s.problem}",
            "",
            f"**Approach.** {s.approach}",
            "",
            f"**Result.** {s.result}",
            "",
            f"**Why it matters.** {s.so_what}",
            "",
        ]
        if s.grounded and s.quote:
            lines += [f"> {s.quote}", ""]
        else:
            lines += [
                "> Citation check failed. The model could not quote the abstract "
                "for its claim, so treat the summary above as unverified.",
                "",
            ]
        if s.reason:
            lines += [f"*Picked because:* {s.reason}", ""]

    unverified = sum(1 for s in summaries if not s.grounded)
    if unverified:
        lines += [
            "---",
            "",
            f"{unverified} of {len(summaries)} summaries failed the citation check.",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def write_digest(text: str, *, out_dir: Path, day: date) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{day.isoformat()}.md"
    path.write_text(text, encoding="utf-8")
    return path

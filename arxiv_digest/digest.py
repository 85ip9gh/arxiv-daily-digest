"""The stored archive, the markdown rendering, and the seen-paper state.

`data/YYYY-MM-DD.json` is the record of record. Markdown and the HTML site are
both rendered from it, so changing a template re-renders every past day instead
of only affecting the next one.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from .agent import Summary

DATA_DIR = "data"
SEEN_FILE = "seen.json"
# Two weeks is long enough that a paper cannot come back through a v2 posting
# in the same fortnight, and short enough that the file stays small.
SEEN_LIMIT = 400


def to_record(summary: Summary) -> dict:
    """Flatten one summary into the shape the archive and the site both read."""
    paper = summary.paper
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": list(paper.authors),
        "author_line": paper.author_line,
        "primary_category": paper.primary_category,
        "categories": list(paper.categories),
        "published": paper.published.isoformat(),
        "abs_url": paper.abs_url,
        "pdf_url": paper.pdf_url,
        "problem": summary.problem,
        "approach": summary.approach,
        "result": summary.result,
        "so_what": summary.so_what,
        "quote": summary.quote,
        "grounded": summary.grounded,
        "reason": summary.reason,
    }


def save_day(
    out_dir: Path, *, day: date, model_label: str, summaries: list[Summary]
) -> Path:
    """Write one day of the archive. This is what the site is rebuilt from."""
    payload = {
        "date": day.isoformat(),
        "model": model_label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "papers": [to_record(s) for s in summaries],
    }
    data_dir = out_dir / DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{day.isoformat()}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_days(out_dir: Path) -> list[dict]:
    """Every archived day, newest first. Unreadable files are skipped, not fatal."""
    data_dir = out_dir / DATA_DIR
    if not data_dir.is_dir():
        return []
    days = []
    for path in sorted(data_dir.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and payload.get("date") and payload.get("papers"):
            days.append(payload)
    return days


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

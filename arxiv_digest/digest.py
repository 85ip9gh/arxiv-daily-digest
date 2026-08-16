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
        "method_details": list(summary.method_details),
        "result": summary.result,
        "numbers": list(summary.numbers),
        "limitations": summary.limitations,
        "so_what": summary.so_what,
        "quote": summary.quote,
        "grounded": summary.grounded,
        "unverified_numbers": list(summary.unverified_numbers),
        "read_full_text": summary.read_full_text,
        "source_label": summary.source_label,
        "reason": summary.reason,
    }


def day_records(out_dir: Path, day: date) -> list[dict]:
    """The papers already archived for one day, or an empty list."""
    path = out_dir / DATA_DIR / f"{day.isoformat()}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    papers = payload.get("papers") if isinstance(payload, dict) else None
    return papers if isinstance(papers, list) else []


def merge_records(existing: list[dict], new: list[dict]) -> list[dict]:
    """Existing papers first, then the new ones, first mention of an id winning.

    Order matters: a second run of the same day is topping up what is there, so
    the morning's papers keep their positions and the additions land after them.
    """
    merged = list(existing)
    known = {r.get("arxiv_id") for r in merged}
    for record in new:
        if record.get("arxiv_id") not in known:
            known.add(record.get("arxiv_id"))
            merged.append(record)
    return merged


def save_day(
    out_dir: Path,
    *,
    day: date,
    model_label: str,
    summaries: list[Summary],
    append: bool = False,
) -> Path:
    """Write one day of the archive. This is what the site is rebuilt from.

    `append` keeps whatever that day already holds. Without it a second run
    replaces the day outright, and because seen.json filters out the papers the
    first run covered, the replacement is a different set: topping a day up
    silently deleted the morning's work.
    """
    records = [to_record(s) for s in summaries]
    if append:
        records = merge_records(day_records(out_dir, day), records)
    payload = {
        "date": day.isoformat(),
        "model": model_label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "papers": records,
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
    return render_records(
        [to_record(s) for s in summaries], day=day, model_label=model_label
    )


def render_records(records: list[dict], *, day: date, model_label: str) -> str:
    """Render the markdown from archive records rather than live summaries.

    Going through the record shape is what lets an append run write markdown
    covering the whole day, including the papers an earlier run produced and
    this process never held as objects.
    """
    count = len(records)
    noun = "paper" if count == 1 else "papers"
    lines = [
        f"# arXiv AI digest, {day.isoformat()}",
        "",
        f"{count} {noun}, summarized by {model_label}.",
        "",
    ]

    for n, r in enumerate(records, start=1):
        lines += [
            f"## {n}. {r['title']}",
            "",
            f"{r['author_line']} | {r['primary_category']} | "
            f"[abstract]({r['abs_url']}) | [pdf]({r['pdf_url']}) | "
            f"read: {r['source_label']}",
            "",
            f"**Problem.** {r['problem']}",
            "",
            f"**Approach.** {r['approach']}",
            "",
        ]
        if r.get("method_details"):
            lines += ["**Method details.**", ""]
            lines += [f"- {d}" for d in r["method_details"]]
            lines += [""]
        lines += [f"**Result.** {r['result']}", ""]
        if r.get("numbers"):
            lines += ["**Numbers.**", ""]
            lines += [f"- {v}" for v in r["numbers"]]
            lines += [""]
        if r.get("limitations"):
            lines += [f"**Limitations.** {r['limitations']}", ""]
        lines += [f"**Why it matters.** {r['so_what']}", ""]

        if r.get("grounded") and r.get("quote"):
            lines += [f"> {r['quote']}", ""]
        else:
            lines += [
                "> Citation check failed. The model could not quote the source "
                "for its claim, so treat the summary above as unverified.",
                "",
            ]
        if r.get("unverified_numbers"):
            lines += [
                "> Figures not found in the source: "
                f"{', '.join(r['unverified_numbers'])}.",
                "",
            ]
        if r.get("reason"):
            lines += [f"*Picked because:* {r['reason']}", ""]

    failures = []
    ungrounded = sum(1 for r in records if not r.get("grounded"))
    if ungrounded:
        failures.append(
            f"{ungrounded} of {count} summaries failed the citation check"
        )
    stray = sum(1 for r in records if r.get("unverified_numbers"))
    if stray:
        failures.append(f"{stray} carry figures that are not in the source")
    if failures:
        lines += ["---", "", ". ".join(failures) + ".", ""]
    return "\n".join(lines).rstrip() + "\n"


def write_digest(text: str, *, out_dir: Path, day: date) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{day.isoformat()}.md"
    path.write_text(text, encoding="utf-8")
    return path

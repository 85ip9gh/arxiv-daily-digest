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
from .contrary import Article
from .hackernews import Story

DATA_DIR = "data"
SEEN_FILE = "seen.json"
SEEN_HN_FILE = "seen-hn.json"
SEEN_CONTRARY_FILE = "seen-contrary.json"
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


def to_story_record(story: Story, reason: str) -> dict:
    """Flatten one picked Hacker News story into the archive's shape.

    Nothing here is verified because nothing here is a claim: the story is
    picked, not summarized, so there is no citation or figure to check
    against a source. The record is just the story's own facts plus the
    selector's one-line reason.
    """
    return {
        "hn_id": story.hn_id,
        "title": story.title,
        "url": story.url,
        "hn_url": story.hn_url,
        "points": story.points,
        "num_comments": story.num_comments,
        "author": story.author,
        "created": story.created.isoformat(),
        "reason": reason,
    }


def to_article_record(article: Article, reason: str) -> dict:
    """Flatten one picked Contrary deep dive into the archive's shape.

    Like a Hacker News story, nothing here is verified because nothing here is
    a claim: the article is picked, not summarized. The record is the article's
    own facts plus the selector's one-line reason.
    """
    return {
        "article_id": article.article_id,
        "title": article.title,
        "url": article.url,
        "authors": list(article.authors),
        "author_line": article.author_line,
        "published": article.published.isoformat(),
        "description": article.description,
        "reason": reason,
    }


def day_records(out_dir: Path, day: date, *, key: str = "papers") -> list[dict]:
    """The records already archived for one day under `key`, or an empty list.

    `key` is `"papers"`, `"stories"` or `"articles"`, the lists a day carries.
    """
    path = out_dir / DATA_DIR / f"{day.isoformat()}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    records = payload.get(key) if isinstance(payload, dict) else None
    return records if isinstance(records, list) else []


def merge_records(
    existing: list[dict], new: list[dict], *, id_key: str = "arxiv_id"
) -> list[dict]:
    """Existing records first, then the new ones, first mention of an id winning.

    Order matters: a second run of the same day is topping up what is there, so
    the morning's records keep their positions and the additions land after
    them. `id_key` is `"arxiv_id"` for papers or `"hn_id"` for stories; the
    default keeps every existing caller working unchanged.
    """
    merged = list(existing)
    known = {r.get(id_key) for r in merged}
    for record in new:
        if record.get(id_key) not in known:
            known.add(record.get(id_key))
            merged.append(record)
    return merged


def save_day(
    out_dir: Path,
    *,
    day: date,
    model_label: str,
    summaries: list[Summary],
    stories: list[tuple[Story, str]] = (),
    articles: list[tuple[Article, str]] = (),
    append: bool = False,
) -> Path:
    """Write one day of the archive. This is what the site is rebuilt from.

    `append` keeps whatever that day already holds. Without it a second run
    replaces the day outright, and because seen.json filters out the papers the
    first run covered, the replacement is a different set: topping a day up
    silently deleted the morning's work. The same logic applies to stories
    against `seen-hn.json` and to articles against `seen-contrary.json`.
    """
    records = [to_record(s) for s in summaries]
    story_records = [to_story_record(s, reason) for s, reason in stories]
    article_records = [to_article_record(a, reason) for a, reason in articles]
    if append:
        records = merge_records(
            day_records(out_dir, day, key="papers"), records, id_key="arxiv_id"
        )
        story_records = merge_records(
            day_records(out_dir, day, key="stories"),
            story_records,
            id_key="hn_id",
        )
        article_records = merge_records(
            day_records(out_dir, day, key="articles"),
            article_records,
            id_key="article_id",
        )
    payload = {
        "date": day.isoformat(),
        "model": model_label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "papers": records,
        "stories": story_records,
        "articles": article_records,
    }
    data_dir = out_dir / DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{day.isoformat()}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_days(out_dir: Path) -> list[dict]:
    """Every archived day, newest first. Unreadable files are skipped, not fatal.

    A day counts as real if it carries papers, stories or articles, so a day
    from a single-source run, or an old day with no `"stories"` or `"articles"`
    key at all, loads the same as one that has all three.
    """
    data_dir = out_dir / DATA_DIR
    if not data_dir.is_dir():
        return []
    days = []
    for path in sorted(data_dir.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and payload.get("date") and (
            payload.get("papers") or payload.get("stories") or payload.get("articles")
        ):
            days.append(payload)
    return days


def load_seen(out_dir: Path, filename: str = SEEN_FILE) -> set[str]:
    path = out_dir / filename
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return set()


def save_seen(
    out_dir: Path, seen: set[str], added: list[str], filename: str = SEEN_FILE
) -> None:
    """Keep insertion order so the oldest ids fall off the end first."""
    ordered = [i for i in added if i not in seen] + sorted(seen)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / filename).write_text(
        json.dumps(ordered[:SEEN_LIMIT], indent=0), encoding="utf-8"
    )


def render(
    summaries: list[Summary],
    *,
    day: date,
    model_label: str,
    stories: list[tuple[Story, str]] = (),
    articles: list[tuple[Article, str]] = (),
) -> str:
    return render_records(
        [to_record(s) for s in summaries],
        day=day,
        model_label=model_label,
        stories=[to_story_record(s, reason) for s, reason in stories],
        articles=[to_article_record(a, reason) for a, reason in articles],
    )


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return singular if n == 1 else (plural or f"{singular}s")


def _join(parts: list[str]) -> str:
    """`a`, `a and b`, or `a, b, and c`. Serial comma from three parts up."""
    if len(parts) <= 1:
        return "".join(parts)
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def render_records(
    records: list[dict],
    *,
    day: date,
    model_label: str,
    stories: list[dict] = (),
    articles: list[dict] = (),
) -> str:
    """Render the markdown from archive records rather than live summaries.

    Going through the record shape is what lets an append run write markdown
    covering the whole day, including the papers an earlier run produced and
    this process never held as objects.
    """
    count = len(records)
    story_count = len(stories)
    article_count = len(articles)

    parts = []
    if count:
        parts.append(f"{count} {_plural(count, 'paper')} summarized")
    if story_count:
        parts.append(
            f"{story_count} Hacker News {_plural(story_count, 'story', 'stories')} picked"
        )
    if article_count:
        parts.append(
            f"{article_count} Contrary Research "
            f"{_plural(article_count, 'deep dive')} picked"
        )
    if parts:
        summary_line = f"{_join(parts)}, by {model_label}."
    else:
        summary_line = f"Nothing published, by {model_label}."

    lines = [
        f"# arXiv AI digest, {day.isoformat()}",
        "",
        summary_line,
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

    if stories:
        lines += ["## From Hacker News", ""]
        for s in stories:
            lines += [
                f"- [{s['title']}]({s['url']}). {s['points']} points, "
                f"{s['num_comments']} comments. {s['reason']} "
                f"[Discussion]({s['hn_url']}).",
            ]
        lines += [""]

    if articles:
        lines += ["## From Contrary Research", ""]
        for a in articles:
            byline = f" by {a['author_line']}" if a.get("author_line") else ""
            lines += [f"- [{a['title']}]({a['url']}).{byline} {a['reason']}"]
        lines += [""]

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

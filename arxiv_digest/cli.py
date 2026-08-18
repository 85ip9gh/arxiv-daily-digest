"""Command line entry point: fetch, select, summarize, archive, publish.

Two independent sources feed one archive. arXiv is read, summarized and
verified; Hacker News is only fetched and picked, since a one-line reason for
a click is not a factual claim that needs a citation. Neither source can take
the other down: each runs behind its own try, logs its own failure, and the
run only exits nonzero when both come back with nothing.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from . import agent, arxiv, digest, hackernews, site
from .llm import LLMConfig, LLMError, RateLimitExhausted, available_models, check

DEFAULT_OUT_DIR = Path("digests")
DEFAULT_SITE_DIR = Path("site")

# Groq's free tier allows 100,000 tokens a day, a number that appears in no
# response header and only in the body of a 429. Selection costs about 6,000 and
# each paper about 4,900, so ten papers sit near 55,000 and leave room for the
# check-failure retries, which re-send a whole paper. Fifteen measured at 79,500
# clean and went over the cap the moment two papers retried.
MAX_COUNT = 10

# Hacker News costs nothing but a selection call, no summarize or verify step,
# so there is no measured token budget behind this number the way there is for
# MAX_COUNT. It is a conservative starting point pending real measurement once
# this runs for a while, the same way the paper count itself was tuned after
# the fact rather than decided up front.
HN_MAX_COUNT = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arxiv-digest",
        description=(
            "Pick a few of the day's new arXiv AI papers and summarize them, "
            "plus a handful of picked Hacker News stories."
        ),
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=3,
        help=f"papers to summarize, capped at {MAX_COUNT} by the daily token budget",
    )
    parser.add_argument(
        "-c",
        "--categories",
        nargs="+",
        default=list(arxiv.DEFAULT_CATEGORIES),
        help="arXiv categories to search",
    )
    parser.add_argument(
        "--hours", type=int, default=48, help="how far back to look for submissions"
    )
    parser.add_argument(
        "--max-results", type=int, default=120, help="candidates to fetch from arXiv"
    )
    parser.add_argument(
        "--shortlist",
        type=int,
        default=40,
        help="how many candidates the selector actually reads",
    )
    parser.add_argument(
        "--interests",
        default=agent.DEFAULT_INTERESTS,
        help="what the selector should favour",
    )
    parser.add_argument(
        "-o", "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="where the archive lives"
    )
    parser.add_argument(
        "-s",
        "--site-dir",
        type=Path,
        default=DEFAULT_SITE_DIR,
        help="where the published HTML lands",
    )
    parser.add_argument(
        "--no-site", action="store_true", help="skip the HTML rebuild"
    )
    parser.add_argument(
        "--no-fulltext",
        action="store_true",
        help="summarize from the abstract only, skipping arXiv's HTML rendering",
    )
    parser.add_argument(
        "--repeats",
        action="store_true",
        help="allow papers that appeared in an earlier digest",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="add to today's archived papers instead of replacing them",
    )
    parser.add_argument(
        "--hn-count",
        type=int,
        default=5,
        help=f"Hacker News stories to pick, capped at {HN_MAX_COUNT}",
    )
    parser.add_argument(
        "--hn-min-points",
        type=int,
        default=60,
        help="minimum points a Hacker News story needs to be a candidate",
    )
    parser.add_argument(
        "--hn-hours",
        type=int,
        default=48,
        help="how far back to look for Hacker News stories",
    )
    parser.add_argument(
        "--hn-interests",
        default=agent.DEFAULT_HN_INTERESTS,
        help="what the Hacker News selector should favour",
    )
    parser.add_argument(
        "--no-hn", action="store_true", help="skip Hacker News entirely"
    )
    parser.add_argument(
        "--hn-repeats",
        action="store_true",
        help="allow stories that appeared in an earlier digest",
    )
    parser.add_argument(
        "--stdout", action="store_true", help="print the digest instead of writing it"
    )
    parser.add_argument(
        "--rebuild-site",
        action="store_true",
        help="regenerate the HTML from the archive and exit, no model calls",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the model backend answers, then exit",
    )
    return parser


def _rebuild(args) -> int:
    days = digest.load_days(args.out_dir)
    if not days:
        print(f"no archived days under {args.out_dir / digest.DATA_DIR}", file=sys.stderr)
        return 1
    written = site.build(days, args.site_dir)
    print(f"rebuilt {len(written)} pages in {args.site_dir}", file=sys.stderr)
    return 0


def _check(config: LLMConfig) -> int:
    try:
        print(f"{check(config)} answered correctly", file=sys.stderr)
        return 0
    except LLMError as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        names = available_models(config)
        if names:
            print(f"models this endpoint offers: {', '.join(names)}", file=sys.stderr)
        return 1


def _run_arxiv(args, config: LLMConfig, seen: set[str]) -> list:
    """Fetch, select and summarize the day's arXiv papers.

    Every failure here is logged and swallowed rather than raised: this
    source must not be able to take Hacker News down with it, so the worst
    this returns is an empty list, never an exception. `seen` is loaded by
    the caller so it is available for `save_seen` later even if the fetch
    below fails.
    """
    try:
        fetched = arxiv.fetch_recent(
            categories=args.categories,
            hours=args.hours,
            max_results=args.max_results,
        )
    except arxiv.FetchError as exc:
        print(f"arxiv: fetch failed: {exc}", file=sys.stderr)
        return []

    count = min(args.count, MAX_COUNT)
    if count < args.count:
        print(
            f"arxiv: asked for {args.count} papers, capping at {MAX_COUNT}: more "
            f"than that does not fit the daily token budget",
            file=sys.stderr,
        )

    candidates = [p for p in fetched.papers if p.arxiv_id not in seen]
    if not candidates:
        print(
            f"arxiv: no unseen papers in the last {fetched.hours}h for "
            f"{', '.join(args.categories)}",
            file=sys.stderr,
        )
        return []

    print(
        f"arxiv: {len(candidates)} candidates from the last {fetched.hours}h, "
        f"selecting {count} with {config.label}",
        file=sys.stderr,
    )

    # Selection is the one call with no partial answer. Everything after it is
    # per paper, so a failure there costs one paper rather than the morning.
    try:
        picks = agent.select(
            candidates,
            config=config,
            count=count,
            interests=args.interests,
            shortlist=args.shortlist,
        )
    except LLMError as exc:
        print(f"arxiv: model error: {exc}", file=sys.stderr)
        return []

    summaries = []
    dropped = []
    for index, (paper, reason) in enumerate(picks):
        print(f"arxiv: summarizing {paper.arxiv_id}: {paper.title[:70]}", file=sys.stderr)
        try:
            summary = agent.summarize(
                paper,
                config=config,
                reason=reason,
                read_body=not args.no_fulltext,
            )
        except RateLimitExhausted as exc:
            # Every remaining paper would fail on the same wall, so stop asking.
            print(f"  out of budget: {exc}", file=sys.stderr)
            dropped.extend(p.arxiv_id for p, _ in picks[index:])
            break
        except LLMError as exc:
            print(f"  skipped: {exc}", file=sys.stderr)
            dropped.append(paper.arxiv_id)
            continue

        flags = [summary.source_label]
        if not summary.grounded:
            flags.append("quote unverified")
        if summary.unverified_numbers:
            flags.append(
                f"{len(summary.unverified_numbers)} figures not in source"
            )
        print(f"  {', '.join(flags)}", file=sys.stderr)
        summaries.append(summary)

    if not summaries:
        print("arxiv: no paper could be summarized", file=sys.stderr)
    elif dropped:
        print(
            f"arxiv: publishing {len(summaries)} of {len(picks)} papers, "
            f"dropped {', '.join(dropped)}",
            file=sys.stderr,
        )
    return summaries


def _run_hn(args, config: LLMConfig, seen: set[str]) -> list:
    """Fetch and pick the day's Hacker News stories.

    No summarize or verify step: the selector's one-line reason is the whole
    "why this is interesting" a reader gets, so there is nothing here to
    check against a source. Same failure contract as `_run_arxiv`: every
    error is logged and swallowed, never raised.
    """
    if args.no_hn:
        return []

    try:
        fetched = hackernews.fetch_recent(
            hours=args.hn_hours, min_points=args.hn_min_points
        )
    except hackernews.FetchError as exc:
        print(f"hn: fetch failed: {exc}", file=sys.stderr)
        return []

    count = min(args.hn_count, HN_MAX_COUNT)
    if count < args.hn_count:
        print(
            f"hn: asked for {args.hn_count} stories, capping at {HN_MAX_COUNT}",
            file=sys.stderr,
        )

    candidates = [s for s in fetched.stories if s.hn_id not in seen]
    if not candidates:
        print(
            f"hn: no unseen stories in the last {fetched.hours}h above "
            f"{args.hn_min_points} points",
            file=sys.stderr,
        )
        return []

    print(
        f"hn: {len(candidates)} candidates from the last {fetched.hours}h, "
        f"picking {count} with {config.label}",
        file=sys.stderr,
    )

    try:
        picks = agent.select_stories(
            candidates,
            config=config,
            count=count,
            interests=args.hn_interests,
            shortlist=args.shortlist,
        )
    except LLMError as exc:
        print(f"hn: model error: {exc}", file=sys.stderr)
        return []

    for story, reason in picks:
        print(f"hn: picked {story.hn_id}: {story.title[:70]}", file=sys.stderr)
    return picks


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # The rebuild path never touches a model, so it must not need a key either.
    if args.rebuild_site:
        return _rebuild(args)

    try:
        config = LLMConfig.from_env()
    except LLMError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.check:
        return _check(config)

    # seen.json and seen-hn.json are read up front, independent of whether
    # either fetch succeeds, so a failed source cannot corrupt the other's
    # dedup state when it is saved back at the end of the run.
    seen = set() if args.repeats else digest.load_seen(args.out_dir)
    hn_seen = (
        set()
        if args.hn_repeats
        else digest.load_seen(args.out_dir, filename=digest.SEEN_HN_FILE)
    )

    # Neither source can take the other down. Each fetches, selects, and
    # (for arXiv) summarizes and verifies behind its own error handling, and
    # only an empty result from both means the run has nothing to publish.
    summaries = _run_arxiv(args, config, seen)
    hn_picks = _run_hn(args, config, hn_seen)

    if not summaries and not hn_picks:
        print(
            f"nothing to publish today, leaving {args.out_dir} untouched",
            file=sys.stderr,
        )
        return 1

    today = date.today()
    # The markdown covers the whole day, so an append run has to render from
    # the merged records rather than from what this process happens to hold.
    records = [digest.to_record(s) for s in summaries]
    story_records = [digest.to_story_record(s, reason) for s, reason in hn_picks]
    if args.append:
        existing_papers = digest.day_records(args.out_dir, today, key="papers")
        records = digest.merge_records(existing_papers, records, id_key="arxiv_id")
        existing_stories = digest.day_records(args.out_dir, today, key="stories")
        story_records = digest.merge_records(
            existing_stories, story_records, id_key="hn_id"
        )
        if existing_papers or existing_stories:
            print(
                f"appending to the {len(existing_papers)} papers and "
                f"{len(existing_stories)} stories already archived for "
                f"{today.isoformat()}",
                file=sys.stderr,
            )
    text = digest.render_records(
        records, day=today, model_label=config.label, stories=story_records
    )

    if args.stdout:
        print(text)
        return 0

    path = digest.write_digest(text, out_dir=args.out_dir, day=today)
    digest.save_day(
        args.out_dir,
        day=today,
        model_label=config.label,
        summaries=summaries,
        stories=hn_picks,
        append=args.append,
    )
    print(f"wrote {path}", file=sys.stderr)

    if not args.repeats:
        digest.save_seen(args.out_dir, seen, [s.paper.arxiv_id for s in summaries])
    if not args.hn_repeats:
        digest.save_seen(
            args.out_dir,
            hn_seen,
            [s.hn_id for s, _ in hn_picks],
            filename=digest.SEEN_HN_FILE,
        )

    if not args.no_site:
        written = site.build(digest.load_days(args.out_dir), args.site_dir)
        print(f"published {len(written)} pages to {args.site_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

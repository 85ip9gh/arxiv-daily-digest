"""Command line entry point: fetch, select, summarize, archive, publish."""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from . import agent, arxiv, digest, site
from .llm import LLMConfig, LLMError, RateLimitExhausted, available_models, check

DEFAULT_OUT_DIR = Path("digests")
DEFAULT_SITE_DIR = Path("site")

# Groq's free tier allows 100,000 tokens a day, a number that appears in no
# response header and only in the body of a 429. Selection costs about 6,000 and
# each paper about 4,900, so ten papers sit near 55,000 and leave room for the
# check-failure retries, which re-send a whole paper. Fifteen measured at 79,500
# clean and went over the cap the moment two papers retried.
MAX_COUNT = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arxiv-digest",
        description="Pick three of the day's new arXiv AI papers and summarize them.",
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

    try:
        fetched = arxiv.fetch_recent(
            categories=args.categories,
            hours=args.hours,
            max_results=args.max_results,
        )
    except arxiv.FetchError as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1

    count = min(args.count, MAX_COUNT)
    if count < args.count:
        print(
            f"asked for {args.count} papers, capping at {MAX_COUNT}: more than "
            f"that does not fit the daily token budget",
            file=sys.stderr,
        )

    seen = set() if args.repeats else digest.load_seen(args.out_dir)
    candidates = [p for p in fetched.papers if p.arxiv_id not in seen]
    if not candidates:
        print(
            f"no unseen papers in the last {fetched.hours}h for "
            f"{', '.join(args.categories)}",
            file=sys.stderr,
        )
        return 1

    print(
        f"{len(candidates)} candidates from the last {fetched.hours}h, "
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
        print(f"model error: {exc}", file=sys.stderr)
        return 1

    summaries = []
    dropped = []
    for index, (paper, reason) in enumerate(picks):
        print(f"summarizing {paper.arxiv_id}: {paper.title[:70]}", file=sys.stderr)
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

    # A short day beats no day. The whole point of the archive is that a run
    # either adds to it or leaves it exactly as it was, and returning here with
    # nothing written is what an empty morning looks like.
    if not summaries:
        print(
            f"no paper could be summarized, leaving {args.out_dir} untouched",
            file=sys.stderr,
        )
        return 1
    if dropped:
        print(
            f"publishing {len(summaries)} of {len(picks)} papers, "
            f"dropped {', '.join(dropped)}",
            file=sys.stderr,
        )

    today = date.today()
    # The markdown covers the whole day, so an append run has to render from the
    # merged records rather than from the summaries this process happens to hold.
    records = [digest.to_record(s) for s in summaries]
    if args.append:
        existing = digest.day_records(args.out_dir, today)
        records = digest.merge_records(existing, records)
        if existing:
            print(
                f"appending {len(records) - len(existing)} to the "
                f"{len(existing)} already archived for {today.isoformat()}",
                file=sys.stderr,
            )
    text = digest.render_records(records, day=today, model_label=config.label)

    if args.stdout:
        print(text)
        return 0

    path = digest.write_digest(text, out_dir=args.out_dir, day=today)
    digest.save_day(
        args.out_dir,
        day=today,
        model_label=config.label,
        summaries=summaries,
        append=args.append,
    )
    print(f"wrote {path}", file=sys.stderr)

    if not args.repeats:
        digest.save_seen(args.out_dir, seen, [s.paper.arxiv_id for s in summaries])

    if not args.no_site:
        written = site.build(digest.load_days(args.out_dir), args.site_dir)
        print(f"published {len(written)} pages to {args.site_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

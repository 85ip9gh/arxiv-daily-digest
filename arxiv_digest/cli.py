"""Command line entry point: fetch, select, summarize, archive, publish."""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from . import agent, arxiv, digest, site
from .llm import LLMConfig, LLMError, available_models, check

DEFAULT_OUT_DIR = Path("digests")
DEFAULT_SITE_DIR = Path("site")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arxiv-digest",
        description="Pick three of the day's new arXiv AI papers and summarize them.",
    )
    parser.add_argument("-n", "--count", type=int, default=3, help="papers to summarize")
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
        "--repeats",
        action="store_true",
        help="allow papers that appeared in an earlier digest",
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
        papers = arxiv.fetch_recent(
            categories=args.categories,
            hours=args.hours,
            max_results=args.max_results,
        )
    except arxiv.FetchError as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1

    seen = set() if args.repeats else digest.load_seen(args.out_dir)
    candidates = [p for p in papers if p.arxiv_id not in seen]
    if not candidates:
        print(
            f"no new papers in the last {args.hours}h for "
            f"{', '.join(args.categories)}",
            file=sys.stderr,
        )
        return 1

    print(
        f"{len(candidates)} candidates, selecting {args.count} with {config.label}",
        file=sys.stderr,
    )

    try:
        picks = agent.select(
            candidates,
            config=config,
            count=args.count,
            interests=args.interests,
            shortlist=args.shortlist,
        )
        summaries = []
        for paper, reason in picks:
            print(f"summarizing {paper.arxiv_id}: {paper.title[:70]}", file=sys.stderr)
            summaries.append(agent.summarize(paper, config=config, reason=reason))
    except LLMError as exc:
        print(f"model error: {exc}", file=sys.stderr)
        return 1

    today = date.today()
    text = digest.render(summaries, day=today, model_label=config.label)

    if args.stdout:
        print(text)
        return 0

    path = digest.write_digest(text, out_dir=args.out_dir, day=today)
    digest.save_day(
        args.out_dir, day=today, model_label=config.label, summaries=summaries
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

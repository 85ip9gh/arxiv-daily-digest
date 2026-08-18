# arxiv-daily-digest

Fetches the day's new AI papers from arXiv, picks a few worth reading, reads
each one's body where arXiv renders it, and publishes technical notes as a
small static site: one page per day, one index you click a date from. A
second, lighter source runs alongside it: a handful of Hacker News stories,
fetched and picked for the same reader, with no summary or verification step
since picking a story makes no factual claim that needs checking.

No paid API, no per-run cost. A local model is still supported for anyone who
would rather nothing left the machine.

## The problem it solves

cs.AI, cs.LG, cs.CL, cs.SE and cs.DC together post more than a hundred papers
on a working day. Reading the titles alone is a daily chore, and a summarizer
that runs unattended at 07:00 is only useful if you can trust what it wrote.

Abstract-only summaries are safe and useless: they restate the marketing
sentence. Notes worth reading name the architecture, the datasets, the
baselines and the measured values, and every one of those is a thing a
summarizer can invent. So the depth comes with two checks.

**The citation.** The model quotes the fragment its result came from, and the
code checks that fragment appears in the source, compared on lowercased words
so punctuation differences do not fail an honest quote.

**The figures.** Every number in the result, the headline figures and the
method details has to appear in the source text. A plausible benchmark score is
the easiest thing in the world to write and the hardest to notice. Prose can be
vague and still honest. A number cannot.

Either failure gets one retry with the specific problem quoted back. If it
fails again the prose survives, the unverifiable quote is dropped, and the page
says which check failed instead of presenting the summary as checked.

The selector's default interests favour a working software engineer's reading
over raw ML novelty: full stack and platform engineering practice, DevOps and
infrastructure automation, agent tooling and verification, and results an
engineer could apply, ahead of pure theory or benchmark chasing. `--interests`
overrides it.

## Hacker News

A second source, run and published alongside arXiv: the Algolia search API
finds recent stories above a point floor, and a selector picks a handful with
a one-line reason each, weighted toward software engineering practice, DevOps
and cloud infrastructure, the tech job market, and notable AI industry news,
away from fundraising announcements, celebrity drama and language holy wars.

There is no summarize or verify step on this side. A summary makes factual
claims that need a citation and a figure check; a one-line reason a story is
worth a click makes none, so there is nothing to check it against. The two
sources are independent: either one failing to fetch or pick anything still
lets the other publish, and the run only fails if both come back empty.

## Install

```bash
git clone https://github.com/85ip9gh/arxiv-daily-digest.git
cd arxiv-daily-digest
pip install -e ".[dev]"
```

Then get a free key. [Groq](https://console.groq.com/keys) is the default and
needs no card:

```bash
set ARXIV_DIGEST_API_KEY=your-key
python -m arxiv_digest --check
```

`--check` runs one trivial completion, so a wrong key or a retired model name
fails in a second instead of at 07:00 tomorrow. If the model name is rejected
it prints the names that endpoint does offer.

Any OpenAI-compatible endpoint works. Google's compatible route, for example:

```bash
set ARXIV_DIGEST_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
set ARXIV_DIGEST_MODEL=gemini-2.5-flash
```

Or run it locally instead, with no key at all:

```bash
set ARXIV_DIGEST_BACKEND=ollama
ollama pull qwen3:8b
```

## Run

```bash
python -m arxiv_digest
```

Writes three things and prints progress to stderr:

| Path | What it is |
|---|---|
| `digests/data/2026-08-15.json` | the archive, and the record of record |
| `digests/2026-08-15.md` | the same day as markdown |
| `site/index.html`, `site/2026-08-15.html` | the published pages |

The site is rebuilt from the whole archive on every run, so a template change
re-renders every past day rather than only the next one. `--rebuild-site` does
that on its own and makes no model calls, so it needs no key.

Useful flags:

| Flag | Default | Does |
|---|---|---|
| `-n, --count` | 3 | papers to summarize |
| `-c, --categories` | cs.AI cs.LG cs.CL cs.SE cs.DC | arXiv categories to search |
| `--hours` | 48 | how far back to look |
| `--shortlist` | 40 | candidates the selector actually reads |
| `--interests` | full stack engineering, DevOps, agent tooling | what the selector favours |
| `--no-fulltext` | off | summarize from the abstract only |
| `--repeats` | off | allow papers from an earlier digest |
| `--append` | off | add to today's archived papers and stories instead of replacing them |
| `--hn-count` | 5 | Hacker News stories to pick, capped at 8 |
| `--hn-min-points` | 60 | minimum points a story needs to be a candidate |
| `--hn-hours` | 48 | how far back to look for stories |
| `--hn-interests` | engineering practice, DevOps, job market, AI news | what the Hacker News selector favours |
| `--no-hn` | off | skip Hacker News entirely |
| `--hn-repeats` | off | allow stories from an earlier digest |
| `--stdout` | off | print instead of writing anything |
| `--no-site` | off | archive and markdown only |
| `-o, --out-dir` | `digests` | where the archive lives |
| `-s, --site-dir` | `site` | where the HTML lands |
| `--rebuild-site` | | regenerate the HTML from the archive, then exit |
| `--check` | | verify the backend answers, then exit |

## The site

Every page is one self-contained HTML file with its CSS and script inline. No
external stylesheet, no font file, no image, nothing fetched at page load, so
the whole thing is a directory any static server hands out read-only. A test
enforces that: the only absolute URLs allowed on a page are the arXiv links a
reader clicks.

The tagline under the masthead counts the papers in the archive rather than
stating a number, because the hardcoded "three" outlived the three paper era by
exactly one config change and then told every visitor the wrong thing. It reads
the largest count in the last seven days, so a short day caused by the token
budget does not misreport the cadence, and a deliberate change to the daily
count is reflected after one run.

`index.html` is a dated ledger, newest first, with a live filter over titles,
categories and dates. A day page reads as a record per paper: problem,
approach, result and why it matters in the open, with method details, the
headline figures and the limitations behind disclosures so the morning skim
stays short without hiding anything. Picked Hacker News stories sit after the
papers as smaller cards: title, points and comments, a link to the discussion,
and the one-line reason it was picked, carrying a `Hacker News` chip so the
two kinds are easy to tell apart at a glance.

The checks are visible where they matter, as chips next to the authors:
`quote verified`, `figures checked`, and what was actually read (`full text` or
`abstract only`). Story cards carry none of these, on purpose: nothing about a
picked story was checked, so nothing claims to be.

Interactive parts, all keyboard reachable: a theme control cycling system,
light and dark; `/` to focus the filter; `j` and `k` between records, papers
and stories alike; `,` and `.` between days; expand or collapse every
disclosure at once; and a copy button per paper. Motion is limited to a 140ms
hover fade and is dropped entirely under `prefers-reduced-motion`.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `ARXIV_DIGEST_BACKEND` | `openai` | or `ollama` to run locally |
| `ARXIV_DIGEST_API_KEY` | none | required unless the backend is `ollama` |
| `ARXIV_DIGEST_MODEL` | `openai/gpt-oss-120b` | `qwen3:8b` under `ollama` |
| `ARXIV_DIGEST_BASE_URL` | `https://api.groq.com/openai/v1` | `http://localhost:11434` under `ollama` |
| `ARXIV_DIGEST_TIMEOUT` | 300 | seconds per call |
| `ARXIV_DIGEST_TEMPERATURE` | 0.2 | |
| `ARXIV_DIGEST_NUM_CTX` | 8192 | Ollama only |

## Daily run on Windows

```powershell
.\install_nightly.ps1
```

Registers a scheduled task at 07:00 logging to `digests\nightly.log`.
`.\install_nightly.ps1 -Remove` unregisters it.

## Daily run on host-b, published on the domain

Live at **https://papers.pesanth.com**.

```bash
bash deploy/install_g7.sh
```

Creates a venv, a `systemd` timer at 07:00 America/Halifax with no spread
running ten papers a day, and an nginx container serving `site/` read-only
under a 64 MB cap,
bound to the tailnet address only. Cloudflare Tunnel is the sole public path,
the same containment every other site on that box uses. Put the API key in
`/home/deploy/arxiv-digest/.env` and the timer starts producing days.

The timezone is spelled out in the unit because the server runs on UTC, where
`07:00` would mean 04:00 in Halifax. `COUNT=3 bash deploy/install_g7.sh` runs a
shorter day; see the token budget section for what each paper costs.

There is deliberately no `RandomizedDelaySec`. A spread start is for fleets that
would stampede a shared service, and this is one box making one arXiv request.
All it bought was a publish time that landed anywhere in a 15 minute window, so
checking at 07:08 could legitimately find yesterday's page. The run takes well
under a minute, so a fixed start puts the day up by 07:01.

Two edge changes are left out of the script on purpose, because their blast
radius is the whole zone rather than this app. Both are already done:

1. The hostname in `/etc/cloudflared/config.yml` above the catch-all 404 rule,
   pointing at `http://100.100.100.100:4245`. Validate before restarting, and note
   the flag goes before the subcommand:
   `cloudflared --config /etc/cloudflared/config.yml tunnel ingress validate`.
2. The DNS record: `cloudflared tunnel route dns <tunnel> <hostname>`.

Recheck every existing hostname after the restart, not only the new one.

The site ships a `robots.txt` that disallows everything. It stays publicly
readable, it just does not ask to be indexed, because machine-written summaries
under a personal domain compete in search with that domain's own pages. Delete
`ROBOTS` in `site.py` to opt back in.

## Things that are load-bearing

**JSON as the record, HTML as a view.** The site is a pure function of
`digests/data/*.json`. That is what makes `--rebuild-site` safe and what stops
a design change from stranding old days in an old layout.

**The body comes from arXiv's HTML rendering**, not the PDF. Most submissions
since late 2023 have one at `arxiv.org/html/<id>`. References, appendices and
LaTeXML math markup are stripped, tables are kept because that is where the
numbers live, and the method and results sections are packed into a 14k
character budget with no single section allowed more than a third of it. A long
method section that swallowed the whole allowance would leave a detailed
mechanism and no measured outcome, which is the wrong half to keep. Papers with
no rendering fall back to the abstract and say so on the page.

**The window widens when the day is thin.** It starts at 48 hours, because
arXiv announces on weekdays only, and doubles up to a week until there are
enough candidates. A Saturday run measured zero papers inside 48 hours and 120
inside 72. Widening costs no extra request: the feed is sorted by date, so a
wider window is just a later cutoff over the response already in hand.

**Long dashes are normalized on the way out.** Papers use en dashes for ranges
and em dashes for asides, and both arrive inside a verbatim quote. The
substitution happens at the boundary rather than in the prompt, where a model
would comply unreliably. Neither check is affected, since both compare on words
and digits rather than punctuation.

**Rate limits are a wait, not a failure.** Free tiers meter tokens per minute,
and three papers of body text in quick succession is exactly the shape that
trips it. A 429 is retried on the provider's own `retry-after`.

**The prompt carries the schema on hosted tiers.** Strict JSON schema support
is uneven across free providers, so the `openai` backend asks for a JSON object
with the schema in the prompt and leans on validation. Ollama constrains
generation with its native `format` schema instead, which is the difference
between usable JSON and a preamble plus prose.

**`num_ctx: 8192` on Ollama.** It defaults to a 4k window and silently
truncates past it. The selection prompt carries dozens of abstracts and does
not fit in 4k. A truncated prompt looks exactly like a bad answer from the
caller's side.

**A 48 hour window, not 24.** arXiv announces on weekdays only, so a Monday run
with a 24 hour window sees an empty weekend.

**Hacker News is picked, not summarized.** A summary makes claims about a
source's contents that need a citation and a figure check. A one-line reason a
story is worth a click makes no such claim, so `select_stories` mirrors
`select()`'s shortlist-then-ask shape with no summarize or verify step behind
it, and no `webtext` fetch of the linked article's body.

## Failure behaviour

Every step degrades instead of crashing, because nobody watches a 07:00 job.
arXiv and Hacker News are independent sources: each is fetched, selected and
(for arXiv) summarized behind its own error handling, so one source failing
never blocks the other from publishing.

- arXiv unreachable: logged, arXiv contributes nothing, Hacker News still
  publishes if it has something.
- Hacker News unreachable or unparseable: logged, Hacker News contributes
  nothing, arXiv still publishes if it has something.
- Both sources unreachable or empty: exit 1, and the existing site is left
  exactly as it was.
- A quiet weekend on arXiv: the window widens to a week before giving up. The
  same widening applies to Hacker News's window.
- arXiv selection unusable after a retry: falls back to the newest papers, and
  says so on the page. Hacker News selection does the same, falling back to
  the newest stories.
- No HTML rendering for a paper: falls back to the abstract, and says so.
- Citation or figures unverifiable after a retry: summary is kept and flagged.
  Hacker News stories are never summarized, so there is nothing to flag.
- Rate limited for seconds: waits the provider's retry-after, up to three times.
- Rate limited for longer than two minutes: stops immediately and publishes the
  papers it already has, rather than napping through a wait it cannot outlast.
- One paper failing: skipped, and the rest of the day is published without it.
- A corrupt archived day is skipped by the site build rather than failing it.

## The token budget

Groq's free tier allows **100,000 tokens a day**, and that number appears in no
response header. The headers advertise a 12,000 token per minute allowance and a
1,000 request per day count, both of which are comfortable. The daily token cap
only shows up in the body of a 429, which makes it easy to misdiagnose as the
per-minute limit it is not.

Measured against the real prompts:

| Item | Tokens |
| --- | --- |
| Selection, 40 candidates | ~6,000 |
| Each paper summarized | ~4,900 |
| Each check-failure retry | +4,900 |

So a day costs roughly `6,000 + 4,900n`, before retries. Three papers is 20,700
and ten is 55,000, which leaves room for the retries that re-send a whole paper.
Fifteen measured at 79,500 clean and went over the cap as soon as two papers
retried, which is why `-n` is capped at ten in `cli.py`.

Two mechanisms keep a run inside this. `TokenWindow` in `llm.py` tracks a 60
second rolling spend and sleeps before a call that would breach the per-minute
allowance, so the pacing is planned rather than discovered through 429s. It
learns the real allowance from `x-ratelimit-limit-tokens`, so a paid tier paces
itself correctly with no code change. The daily cap cannot be paced around, so
hitting it stops the run and publishes what is already summarized.

Papers already summarized are recorded in `digests/seen.json` and skipped, so
the same paper does not lead the digest two days running. Hacker News stories
are recorded the same way in `digests/seen-hn.json`, a separate file so a
paper's arXiv id and a story's HN id can never collide. Hacker News costs one
selection call and no per-paper summarize or retry, so it barely moves this
budget; `--hn-count` is capped at `HN_MAX_COUNT` in `cli.py` as a conservative
starting point pending real measurement once it has run for a while, the same
way the paper count itself was tuned after the fact rather than decided up
front.

That filter is also why a second run of the same day needs `--append`. Without
it the day is rewritten from scratch, and since the first run's papers (and
stories) are now in the seen files the rewrite picks a different set, deleting
the morning's work rather than adding to it. With `--append` the existing
papers and stories keep their positions and the new ones land after them, so
a top-up run that dies partway can only leave the day the same or longer.

## Tests

```bash
python -m pytest
```

190 tests, no network and no model. The arXiv parser runs against a fixture
feed, the full-text reader against a fixture rendering, the Hacker News client
against canned Algolia responses, and the agent tests replace the model call
with canned responses including the invented quotes and the invented benchmark
scores the two checks exist to catch. The site tests cover escaping, the
pager, the theme tokens (every token defined on bare `:root`, both explicit
themes redefining the same set), story cards, and the promise that a page
fetches nothing at load. The CLI tests cover both sources failing
independently, so one source going down never takes the other with it.

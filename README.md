# arxiv-daily-digest

Fetches the day's new AI papers from arXiv, picks three worth reading,
summarizes them with a hosted free-tier model, and publishes the result as a
small static site: one page per day, one index you click a date from.

No paid API, no per-run cost. A local model is still supported for anyone who
would rather nothing left the machine.

## The problem it solves

cs.AI, cs.LG and cs.CL together post more than a hundred papers on a working
day. Reading the titles alone is a daily chore, and a summarizer that runs
unattended at 07:00 is only useful if you can trust what it wrote.

A model asked to summarize a paper it half-recognizes will describe the paper
it remembers rather than the one in front of it, and that failure is invisible
in fluent prose. So every summary has to quote the abstract:

1. The model returns four short fields plus the fragment its result came from.
2. The code checks that fragment actually appears in the abstract, compared on
   lowercased words so punctuation differences do not fail an honest citation.
3. A quote that fails gets one retry with the failure quoted back.
4. If it fails again, **the quote is discarded and the summary is marked
   unverified**, on the page and in the markdown, rather than printed as though
   it had been checked.

The prose survives either way. The claim to have read the paper does not.

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
| `-c, --categories` | cs.AI cs.LG cs.CL | arXiv categories to search |
| `--hours` | 48 | how far back to look |
| `--shortlist` | 40 | candidates the selector actually reads |
| `--interests` | agents, retrieval, evals, small models | what the selector favours |
| `--repeats` | off | allow papers from an earlier digest |
| `--stdout` | off | print instead of writing anything |
| `--no-site` | off | archive and markdown only |
| `-o, --out-dir` | `digests` | where the archive lives |
| `-s, --site-dir` | `site` | where the HTML lands |
| `--rebuild-site` | | regenerate the HTML from the archive, then exit |
| `--check` | | verify the backend answers, then exit |

## The site

Every page is one self-contained HTML file with its CSS inline. No scripts, no
fonts, no images, nothing fetched at page load, so the whole thing is a
directory any static server can hand out read-only. It follows the visitor's
light or dark preference.

`index.html` lists every day newest first with its three titles under it. A day
page carries the four fields, the checked quote, the selector's reason for
picking each paper, links to the abstract and the pdf, and older/newer
navigation.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `ARXIV_DIGEST_BACKEND` | `openai` | or `ollama` to run locally |
| `ARXIV_DIGEST_API_KEY` | none | required unless the backend is `ollama` |
| `ARXIV_DIGEST_MODEL` | `llama-3.3-70b-versatile` | `qwen3:8b` under `ollama` |
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

Creates a venv, a `systemd` timer at 07:00 America/Halifax with a 15 minute
spread, and an nginx container serving `site/` read-only under a 64 MB cap,
bound to the tailnet address only. Cloudflare Tunnel is the sole public path,
the same containment every other site on that box uses. Put the API key in
`/home/deploy/arxiv-digest/.env` and the timer starts producing days.

The timezone is spelled out in the unit because the server runs on UTC, where
`07:00` would mean 04:00 in Halifax.

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

## Four things that are load-bearing

**JSON as the record, HTML as a view.** The site is a pure function of
`digests/data/*.json`. That is what makes `--rebuild-site` safe and what stops
a design change from stranding old days in an old layout.

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

## Failure behaviour

Every step degrades instead of crashing, because nobody watches a 07:00 job.

- arXiv unreachable: two retries with a delay, then exit 1 with the reason.
- Selection unusable after a retry: falls back to the three newest papers, and
  says so on the page.
- Citation unverifiable after a retry: summary is kept and flagged.
- Model unreachable: exit 1, and the existing site is left exactly as it was.
- A corrupt archived day is skipped by the site build rather than failing it.

Papers already summarized are recorded in `digests/seen.json` and skipped, so
the same paper does not lead the digest two days running.

## Tests

```bash
python -m pytest
```

56 tests, no network and no model. The arXiv parser runs against a fixture
feed, the agent tests replace the model call with canned responses including
the invented quotes the citation check exists to catch, and the site tests
cover escaping, the older/newer pager, and the promise that a page fetches
nothing at load.

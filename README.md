# arxiv-daily-digest

Fetches the day's new AI papers from arXiv, picks three worth reading, and
summarizes them with a local model. One markdown file per day.

Runs on one consumer GPU. No API key, no per-run cost, no data leaving the
machine. A hosted free tier is supported for anyone without a GPU.

## The problem it solves

cs.AI, cs.LG and cs.CL together post more than a hundred papers on a working
day. Reading the titles alone is a daily chore, and a summarizer that runs
unattended at 07:00 is only useful if you can trust what it wrote.

A small model asked to summarize a paper it half-recognizes will describe the
paper it remembers rather than the one in front of it, and that failure is
invisible in fluent prose. So every summary has to quote the abstract:

1. The model returns four short fields plus the fragment its result came from.
2. The code checks that fragment actually appears in the abstract, compared on
   lowercased words so punctuation differences do not fail an honest citation.
3. A quote that fails gets one retry with the failure quoted back.
4. If it fails again, **the quote is discarded and the summary is marked
   unverified in the digest**, rather than printed as though it were checked.

The prose survives either way. The claim to have read the paper does not.

## Install

```bash
git clone https://github.com/85ip9gh/arxiv-daily-digest.git
cd arxiv-daily-digest
pip install -e ".[dev]"
```

Then either run a local model, which is the default:

```bash
ollama pull qwen3:8b
```

or point it at a free hosted tier:

```bash
set ARXIV_DIGEST_BACKEND=openai
set ARXIV_DIGEST_API_KEY=your-key
set ARXIV_DIGEST_MODEL=llama-3.3-70b-versatile
```

## Run

```bash
python -m arxiv_digest
```

Writes `digests/2026-08-15.md` and prints progress to stderr. A full run on an
RTX 5060 (8 GB) took 68 seconds: one selection pass over 27 candidates and
three summaries.

Useful flags:

| Flag | Default | Does |
|---|---|---|
| `-n, --count` | 3 | papers to summarize |
| `-c, --categories` | cs.AI cs.LG cs.CL | arXiv categories to search |
| `--hours` | 48 | how far back to look |
| `--shortlist` | 40 | candidates the selector actually reads |
| `--interests` | agents, retrieval, evals, small models | what the selector favours |
| `--repeats` | off | allow papers from an earlier digest |
| `--stdout` | off | print instead of writing a file |
| `-o, --out-dir` | `digests` | where digests land |

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `ARXIV_DIGEST_BACKEND` | `ollama` | or `openai` for any OpenAI-compatible endpoint |
| `ARXIV_DIGEST_MODEL` | `qwen3:8b` | |
| `ARXIV_DIGEST_BASE_URL` | `http://localhost:11434` | Groq, OpenRouter and Google's compatible route all work |
| `ARXIV_DIGEST_API_KEY` | none | required by the `openai` backend only |
| `ARXIV_DIGEST_NUM_CTX` | 8192 | Ollama only |
| `ARXIV_DIGEST_TIMEOUT` | 300 | seconds per call |
| `ARXIV_DIGEST_TEMPERATURE` | 0.2 | |

## Daily run

```powershell
.\install_nightly.ps1
```

Registers a Windows scheduled task at 07:00 that runs the digest and logs to
`digests\nightly.log`. `.\install_nightly.ps1 -Remove` unregisters it.

## Four things that are load-bearing

**Ollama's `format` schema.** Without a schema constraint an 8B model adds a
preamble and drifts field names often enough to be unusable. The `openai`
backend cannot rely on this, because strict JSON schema support is uneven
across free providers, so it asks for a JSON object with the schema in the
prompt and leans harder on validation.

**`think: false`.** Qwen3 reasons before answering by default. Restating an
abstract has no reasoning in it, and the thinking tokens are pure latency on an
8 GB card.

**`num_ctx: 8192`.** Ollama defaults to a 4k window and silently truncates past
it. The selection prompt carries dozens of abstracts and does not fit in 4k. A
truncated prompt looks exactly like a bad answer from the caller's side.

**A 48 hour window, not 24.** arXiv announces on weekdays only, so a Monday run
with a 24 hour window sees an empty weekend.

## Failure behaviour

Every step degrades instead of crashing, because nobody watches a 07:00 job.

- arXiv unreachable: two retries with a delay, then exit 1 with the reason.
- Selection unusable after a retry: falls back to the three newest papers, and
  says so in the digest.
- Citation unverifiable after a retry: summary is kept and flagged.
- Model unreachable: exit 1, no half-written digest.

Papers already summarized are recorded in `digests/seen.json` and skipped, so
the same paper does not lead the digest two days running.

## Tests

```bash
python -m pytest
```

40 tests, no network and no model. The arXiv parser runs against a fixture
feed, and the agent tests replace the model call with canned responses,
including the invented quotes the citation check exists to catch.

"""One JSON-returning `complete` call, over either a hosted or a local model.

Two backends, same function signature:

`openai`  the default. Any OpenAI-compatible endpoint, which covers the free
          tiers worth using (Groq, Google's compatible route, OpenRouter).
          Support for strict JSON schemas is uneven across those providers, so
          this backend asks for a JSON object and puts the schema in the prompt
          instead. That is weaker, hence the validation in `agent.py`.

`ollama`  a local daemon, opt in with `ARXIV_DIGEST_BACKEND=ollama`. Constrains
          generation with Ollama's native `format` schema, which is what makes
          an 8B model return usable JSON instead of a preamble plus prose.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

import requests

DEFAULT_BACKEND = "openai"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
# Summarizing a 1,500 character abstract on a consumer GPU runs well under a
# minute, but a cold model load is the slow part of the first call of the day.
DEFAULT_TIMEOUT = 300

# Groq's free tier meters 12,000 tokens per minute on llama-3.3-70b-versatile,
# read off x-ratelimit-limit-tokens. A full-text paper is roughly 4,500 of them,
# so about two and a half papers fit in a minute and everything past that is a
# 429. The real limit is learned from the response headers on the first call;
# this is only the opening guess.
DEFAULT_TOKENS_PER_MINUTE = 12000
RATE_WINDOW = 60.0
# What a summarize call spends beyond its prompt. Deliberately generous: guessing
# high costs a little wall clock, guessing low costs the day.
COMPLETION_ALLOWANCE = 1200
# Tokens per character of prompt. English prose sits near a quarter.
CHARS_PER_TOKEN = 4


# The longest 429 wait worth sitting through. A per-minute limit clears in
# seconds; a per-day one comes back asking for half an hour, and waiting that
# out at 07:00 is not a run, it is a hang. Past this the run stops and keeps
# whatever it already has.
MAX_AFFORDABLE_WAIT = 120.0


class LLMError(RuntimeError):
    """The model call failed, or returned something that was not valid JSON."""


class RateLimitExhausted(LLMError):
    """The allowance is gone for long enough that waiting is not an option.

    Separate from LLMError because the callers want different things. One paper
    failing to summarize is worth skipping past; the day's token budget running
    out means every remaining paper will fail the same way, so the run should
    stop and publish what it has.
    """


@dataclass(frozen=True)
class LLMConfig:
    backend: str = DEFAULT_BACKEND
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    timeout: int = DEFAULT_TIMEOUT
    temperature: float = 0.2
    # Ollama defaults to a 4k window and silently truncates past it. The
    # selection prompt carries dozens of abstracts and does not fit in 4k, and
    # a truncated prompt looks exactly like a bad answer from the caller's side.
    num_ctx: int = 8192
    # Three is enough once the pacing below keeps the run inside the per-minute
    # allowance. Raising it does not help: the failure that actually happens is
    # the daily cap, and no number of 90 second naps clears a 32 minute wait.
    rate_limit_retries: int = 3
    # 0 disables pacing, which is right for a local daemon that meters nothing.
    tokens_per_minute: int = DEFAULT_TOKENS_PER_MINUTE

    @classmethod
    def from_env(cls) -> "LLMConfig":
        backend = os.environ.get("ARXIV_DIGEST_BACKEND", DEFAULT_BACKEND).strip().lower()
        if backend not in {"ollama", "openai"}:
            raise LLMError(f"unknown backend {backend!r}, expected openai or ollama")

        if backend == "ollama":
            model = os.environ.get("ARXIV_DIGEST_MODEL", DEFAULT_OLLAMA_MODEL)
            base_url = os.environ.get("ARXIV_DIGEST_BASE_URL", DEFAULT_OLLAMA_URL)
            api_key = None
        else:
            model = os.environ.get("ARXIV_DIGEST_MODEL", DEFAULT_MODEL)
            base_url = os.environ.get("ARXIV_DIGEST_BASE_URL", DEFAULT_BASE_URL)
            api_key = os.environ.get("ARXIV_DIGEST_API_KEY")
            if not api_key:
                raise LLMError(
                    "backend openai needs ARXIV_DIGEST_API_KEY in the environment"
                )

        default_tpm = 0 if backend == "ollama" else DEFAULT_TOKENS_PER_MINUTE
        return cls(
            backend=backend,
            model=model,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=int(os.environ.get("ARXIV_DIGEST_TIMEOUT", DEFAULT_TIMEOUT)),
            temperature=float(os.environ.get("ARXIV_DIGEST_TEMPERATURE", "0.2")),
            num_ctx=int(os.environ.get("ARXIV_DIGEST_NUM_CTX", "8192")),
            tokens_per_minute=int(
                os.environ.get("ARXIV_DIGEST_TOKENS_PER_MINUTE", default_tpm)
            ),
        )

    @property
    def label(self) -> str:
        if self.backend == "ollama":
            return f"{self.model} (local)"
        host = urllib.parse.urlparse(self.base_url).hostname or self.base_url
        return f"{self.model} ({host})"


class TokenWindow:
    """Spends a tokens-per-minute allowance on purpose instead of by 429.

    The old loop fired calls back to back and let the rate limit sort it out.
    That works at three papers and dies at fifteen, because backoff is reactive:
    by the time the 429 arrives the minute is already spent, and a paper that
    needs more waits than the retry budget kills the whole run.

    This keeps a 60 second window of what has been spent and sleeps before a
    call that would not fit. Wall clock at 07:00 is free; a missing digest is
    not. `now` and `sleep` are injected so the tests do not run in real time.
    """

    def __init__(self, now=time.monotonic, sleep=time.sleep) -> None:
        self._now = now
        self._sleep = sleep
        self._spent: list[tuple[float, int]] = []
        self.limit = 0

    def _prune(self, now: float) -> None:
        cutoff = now - RATE_WINDOW
        self._spent = [(t, n) for t, n in self._spent if t > cutoff]

    def reserve(self, tokens: int) -> float:
        """Wait until `tokens` fit in the window, then book them. Returns seconds slept.

        A single request larger than the whole allowance is let through rather
        than looped on forever. The provider will answer that one with a 429 and
        the retry path handles it, which is the correct division of labour.
        """
        if self.limit <= 0:
            return 0.0
        waited = 0.0
        while True:
            now = self._now()
            self._prune(now)
            used = sum(n for _, n in self._spent)
            if not self._spent or used + tokens <= self.limit:
                self._spent.append((now, tokens))
                return waited
            nap = max(0.5, RATE_WINDOW - (now - self._spent[0][0]))
            self._sleep(nap)
            waited += nap

    def settle(self, actual: int) -> None:
        """Replace the last estimate with what the response says it really cost."""
        if self._spent and actual > 0:
            when, _ = self._spent[-1]
            self._spent[-1] = (when, actual)


_WINDOW = TokenWindow()


def _estimate_tokens(body: dict[str, Any]) -> int:
    chars = sum(len(str(m.get("content", ""))) for m in body.get("messages", []))
    return chars // CHARS_PER_TOKEN + COMPLETION_ALLOWANCE


def _learn_limit(response, config: LLMConfig) -> None:
    """Take the real allowance from the response rather than trusting the constant.

    Costs nothing, and means an upgraded tier or a different provider paces
    itself correctly without a code change.
    """
    raw = response.headers.get("x-ratelimit-limit-tokens")
    try:
        reported = int(str(raw).strip())
    except (TypeError, ValueError):
        return
    if reported > 0:
        _WINDOW.limit = min(reported, config.tokens_per_minute) if config.tokens_per_minute else reported


def _post(url: str, body: dict[str, Any], config: LLMConfig) -> dict[str, Any]:
    """One call, paced to the token allowance, with a 429 treated as a wait.

    Two layers, and they do different jobs. `TokenWindow` keeps the run inside
    the allowance so the 429 mostly never happens. The retry loop is what
    catches the cases the estimate got wrong, which is why both are here.
    """
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    _WINDOW.limit = _WINDOW.limit or config.tokens_per_minute
    estimate = _estimate_tokens(body)

    for attempt in range(config.rate_limit_retries + 1):
        _WINDOW.reserve(estimate)
        try:
            response = requests.post(
                url, json=body, headers=headers, timeout=config.timeout
            )
        except requests.RequestException as exc:
            raise LLMError(f"{config.model} call failed: {exc}") from exc

        _learn_limit(response, config)

        if response.status_code == 429:
            wait = _retry_after(response)
            # A per-day cap answers with a wait measured in half hours. Grinding
            # through the retry budget against that burns twelve minutes and
            # fails anyway, so say what the provider said and let the caller
            # keep what it has.
            if wait > MAX_AFFORDABLE_WAIT:
                raise RateLimitExhausted(
                    f"{config.model} is rate limited for {wait:.0f}s, longer than "
                    f"this run can wait: {_limit_reason(response)}"
                )
            if attempt < config.rate_limit_retries:
                time.sleep(wait)
                continue
        if response.status_code >= 400:
            raise LLMError(
                f"{config.model} call failed: HTTP {response.status_code} "
                f"body={response.text[:200]!r}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMError(f"{config.model} returned no JSON body") from exc

        usage = payload.get("usage") if isinstance(payload, dict) else None
        if isinstance(usage, dict):
            try:
                _WINDOW.settle(int(usage.get("total_tokens", 0)))
            except (TypeError, ValueError):
                pass
        return payload

    raise LLMError(f"{config.model} stayed rate limited after {config.rate_limit_retries} waits")


def _limit_reason(response) -> str:
    """Pull the provider's own explanation out of the 429.

    Worth the few lines: the daily cap is invisible in the rate limit headers,
    which advertise only the per-minute token allowance and the daily request
    count. The body is the one place it says "tokens per day (TPD)", and without
    it the log blames the wrong limit.
    """
    try:
        message = response.json().get("error", {}).get("message", "")
    except (ValueError, AttributeError):
        message = ""
    return str(message).strip()[:300] or "the provider gave no reason"


def _retry_after(response) -> float:
    raw = response.headers.get("retry-after") or response.headers.get(
        "x-ratelimit-reset-tokens", ""
    )
    try:
        return max(1.0, float(str(raw).rstrip("s")))
    except (TypeError, ValueError):
        return 20.0


def _parse_json(content: str, config: LLMConfig) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"{config.model} returned non-JSON: {content[:200]!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise LLMError(f"{config.model} returned {type(parsed).__name__}, not an object")
    return parsed


def available_models(config: LLMConfig) -> list[str]:
    """Best effort list of model ids, for when a configured name is rejected.

    Free tiers retire model names without much notice, and "model not found"
    is the one failure where the fix is a name the provider will hand over.
    """
    try:
        if config.backend == "ollama":
            payload = requests.get(f"{config.base_url}/api/tags", timeout=30).json()
            return sorted(str(m.get("name", "")) for m in payload.get("models", []))
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        payload = requests.get(
            f"{config.base_url}/models", headers=headers, timeout=30
        ).json()
        return sorted(str(m.get("id", "")) for m in payload.get("data", []))
    except (requests.RequestException, ValueError, AttributeError):
        return []


def check(config: LLMConfig) -> str:
    """Run one trivial call so a bad key or model name fails now, not at 07:00."""
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    answer = complete(
        'Reply with {"ok": true} and nothing else.', schema, config=config
    )
    if "ok" not in answer:
        raise LLMError(f"{config.label} answered {answer!r}, which has no ok field")
    return config.label


def complete(
    prompt: str,
    schema: dict[str, Any],
    *,
    config: LLMConfig,
    system: str | None = None,
) -> dict[str, Any]:
    """Run one completion and return the parsed JSON object."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})

    if config.backend == "ollama":
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": config.model,
            "messages": messages,
            "format": schema,
            "stream": False,
            # Qwen3 and the DeepSeek distills reason before answering by default.
            # Reading an abstract and restating it has no reasoning in it, and the
            # thinking tokens are pure latency on an 8 GB card.
            "think": False,
            "options": {
                "temperature": config.temperature,
                "num_ctx": config.num_ctx,
            },
        }
        payload = _post(f"{config.base_url}/api/chat", body, config)
        content = payload.get("message", {}).get("content", "")
    else:
        schema_text = json.dumps(schema, indent=2)
        messages.append(
            {
                "role": "user",
                "content": f"{prompt}\n\nReturn one JSON object matching this schema, and nothing else:\n{schema_text}",
            }
        )
        body = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "response_format": {"type": "json_object"},
        }
        payload = _post(f"{config.base_url}/chat/completions", body, config)
        choices = payload.get("choices") or []
        if not choices:
            raise LLMError(f"{config.model} returned no choices")
        content = choices[0].get("message", {}).get("content", "")

    return _parse_json(content, config)

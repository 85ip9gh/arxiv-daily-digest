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


class LLMError(RuntimeError):
    """The model call failed, or returned something that was not valid JSON."""


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

        return cls(
            backend=backend,
            model=model,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=int(os.environ.get("ARXIV_DIGEST_TIMEOUT", DEFAULT_TIMEOUT)),
            temperature=float(os.environ.get("ARXIV_DIGEST_TEMPERATURE", "0.2")),
            num_ctx=int(os.environ.get("ARXIV_DIGEST_NUM_CTX", "8192")),
        )

    @property
    def label(self) -> str:
        if self.backend == "ollama":
            return f"{self.model} (local)"
        host = urllib.parse.urlparse(self.base_url).hostname or self.base_url
        return f"{self.model} ({host})"


def _post(url: str, body: dict[str, Any], config: LLMConfig) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    try:
        response = requests.post(
            url, json=body, headers=headers, timeout=config.timeout
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" body={exc.response.text[:200]!r}"
        raise LLMError(f"{config.model} call failed: {exc}{detail}") from exc
    return response.json()


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

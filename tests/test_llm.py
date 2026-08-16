import json

import pytest

from arxiv_digest import llm
from arxiv_digest.llm import LLMConfig, LLMError

CONFIG = LLMConfig(
    backend="ollama", model="qwen3:8b", base_url="http://localhost:11434"
)


class TestParsing:
    def test_plain_object(self):
        assert llm._parse_json('{"a": 1}', CONFIG) == {"a": 1}

    def test_fenced_object(self):
        assert llm._parse_json('```json\n{"a": 1}\n```', CONFIG) == {"a": 1}

    def test_object_with_a_preamble(self):
        text = 'Sure, here is the summary:\n{"a": 1}\nHope that helps.'
        assert llm._parse_json(text, CONFIG) == {"a": 1}

    def test_prose_raises(self):
        with pytest.raises(LLMError):
            llm._parse_json("I could not read the abstract.", CONFIG)

    def test_bare_array_raises(self):
        with pytest.raises(LLMError):
            llm._parse_json("[1, 2, 3]", CONFIG)


class TestConfig:
    def _clear(self, monkeypatch):
        for name in [
            "ARXIV_DIGEST_BACKEND",
            "ARXIV_DIGEST_MODEL",
            "ARXIV_DIGEST_BASE_URL",
            "ARXIV_DIGEST_API_KEY",
        ]:
            monkeypatch.delenv(name, raising=False)

    def test_a_hosted_free_tier_is_the_default(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("ARXIV_DIGEST_API_KEY", "test-key")
        config = LLMConfig.from_env()
        assert config.backend == "openai"
        assert config.base_url == "https://api.groq.com/openai/v1"
        assert config.label.endswith("(api.groq.com)")

    def test_ollama_stays_available_by_name(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("ARXIV_DIGEST_BACKEND", "ollama")
        config = LLMConfig.from_env()
        assert config.model == "qwen3:8b"
        assert config.api_key is None
        assert config.label.endswith("(local)")

    def test_the_default_backend_needs_a_key(self, monkeypatch):
        self._clear(monkeypatch)
        with pytest.raises(LLMError):
            LLMConfig.from_env()

    def test_openai_backend_strips_a_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("ARXIV_DIGEST_BACKEND", "openai")
        monkeypatch.setenv("ARXIV_DIGEST_API_KEY", "test-key")
        monkeypatch.setenv("ARXIV_DIGEST_BASE_URL", "https://example.invalid/v1/")
        assert LLMConfig.from_env().base_url == "https://example.invalid/v1"

    def test_unknown_backend_raises(self, monkeypatch):
        monkeypatch.setenv("ARXIV_DIGEST_BACKEND", "anthropic")
        with pytest.raises(LLMError):
            LLMConfig.from_env()


class TestRequestShape:
    def _capture(self, monkeypatch):
        seen = {}

        def fake_post(url, body, config):
            seen["url"] = url
            seen["body"] = body
            if config.backend == "ollama":
                return {"message": {"content": '{"ok": true}'}}
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

        monkeypatch.setattr(llm, "_post", fake_post)
        return seen

    def test_ollama_constrains_generation_with_the_schema(self, monkeypatch):
        seen = self._capture(monkeypatch)
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        assert llm.complete("hi", schema, config=CONFIG) == {"ok": True}
        assert seen["url"].endswith("/api/chat")
        assert seen["body"]["format"] == schema
        assert seen["body"]["think"] is False

    def test_openai_puts_the_schema_in_the_prompt(self, monkeypatch):
        seen = self._capture(monkeypatch)
        config = LLMConfig(
            backend="openai",
            model="llama-3.3-70b-versatile",
            base_url="https://example.invalid/v1",
            api_key="test-key",
        )
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        assert llm.complete("hi", schema, config=config) == {"ok": True}
        assert seen["url"].endswith("/chat/completions")
        assert seen["body"]["response_format"] == {"type": "json_object"}
        assert json.dumps(schema, indent=2) in seen["body"]["messages"][-1]["content"]

    def test_system_prompt_is_passed_through(self, monkeypatch):
        seen = self._capture(monkeypatch)
        llm.complete("hi", {"type": "object"}, config=CONFIG, system="be terse")
        assert seen["body"]["messages"][0] == {"role": "system", "content": "be terse"}


class FakeClock:
    """A monotonic clock that only moves when something sleeps."""

    def __init__(self):
        self.t = 1000.0
        self.naps = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.naps.append(seconds)
        self.t += seconds


class TestTokenWindow:
    def test_calls_inside_the_allowance_never_wait(self):
        clock = FakeClock()
        window = llm.TokenWindow(now=clock.now, sleep=clock.sleep)
        window.limit = 12000
        for _ in range(2):
            assert window.reserve(4500) == 0.0
        assert clock.naps == []

    def test_the_call_that_would_break_the_limit_waits_first(self):
        clock = FakeClock()
        window = llm.TokenWindow(now=clock.now, sleep=clock.sleep)
        window.limit = 12000
        window.reserve(4500)
        window.reserve(4500)
        # 9,000 spent, so a third 4,500 would put the minute at 13,500.
        assert window.reserve(4500) > 0
        assert clock.naps, "expected a sleep before the third call"

    def test_the_wait_is_only_as_long_as_the_window(self):
        clock = FakeClock()
        window = llm.TokenWindow(now=clock.now, sleep=clock.sleep)
        window.limit = 12000
        window.reserve(11000)
        waited = window.reserve(4500)
        assert 0 < waited <= llm.RATE_WINDOW

    def test_spend_leaves_the_window_after_a_minute(self):
        clock = FakeClock()
        window = llm.TokenWindow(now=clock.now, sleep=clock.sleep)
        window.limit = 12000
        window.reserve(11000)
        clock.t += llm.RATE_WINDOW + 1
        assert window.reserve(11000) == 0.0

    def test_a_request_bigger_than_the_allowance_is_let_through(self):
        """Otherwise it loops forever. The 429 path is the right owner for this."""
        clock = FakeClock()
        window = llm.TokenWindow(now=clock.now, sleep=clock.sleep)
        window.limit = 12000
        assert window.reserve(50000) == 0.0

    def test_no_limit_means_no_pacing(self):
        clock = FakeClock()
        window = llm.TokenWindow(now=clock.now, sleep=clock.sleep)
        for _ in range(10):
            assert window.reserve(99999) == 0.0
        assert clock.naps == []

    def test_settle_replaces_the_estimate_with_the_real_cost(self):
        clock = FakeClock()
        window = llm.TokenWindow(now=clock.now, sleep=clock.sleep)
        window.limit = 12000
        window.reserve(4000)
        window.settle(11500)
        # The estimate said 4,000 and the truth was 11,500, so the next call waits.
        assert window.reserve(4000) > 0


class TestPacingDefaults:
    def test_hosted_backend_paces_by_default(self, monkeypatch):
        for name in ("ARXIV_DIGEST_BACKEND", "ARXIV_DIGEST_TOKENS_PER_MINUTE"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("ARXIV_DIGEST_API_KEY", "test-key")
        assert LLMConfig.from_env().tokens_per_minute == llm.DEFAULT_TOKENS_PER_MINUTE

    def test_ollama_does_not_pace(self, monkeypatch):
        monkeypatch.delenv("ARXIV_DIGEST_TOKENS_PER_MINUTE", raising=False)
        monkeypatch.setenv("ARXIV_DIGEST_BACKEND", "ollama")
        assert LLMConfig.from_env().tokens_per_minute == 0

    def test_the_retry_budget_stays_small(self):
        """Pacing is what keeps the run legal. Retries only catch a bad estimate."""
        assert LLMConfig().rate_limit_retries == 3

    def test_estimate_counts_the_prompt_and_allows_for_the_answer(self):
        body = {"messages": [{"role": "user", "content": "x" * 4000}]}
        assert llm._estimate_tokens(body) == 1000 + llm.COMPLETION_ALLOWANCE


class FakeResponse:
    def __init__(self, status_code=429, headers=None, payload=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload if payload is not None else {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


TPD_BODY = {
    "error": {
        "message": (
            "Rate limit reached for model `llama-3.3-70b-versatile` on tokens "
            "per day (TPD): Limit 100000, Used 98625, Requested 3586."
        ),
        "code": "rate_limit_exceeded",
    }
}


class TestDailyCapFailsFast:
    """The daily cap is the failure that actually happens, and it is not a wait."""

    def _post_returning(self, monkeypatch, response, slept):
        monkeypatch.setattr(llm.requests, "post", lambda *a, **k: response)
        monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))
        monkeypatch.setattr(llm._WINDOW, "limit", 0)

    def test_a_half_hour_wait_raises_instead_of_napping(self, monkeypatch):
        slept = []
        response = FakeResponse(headers={"retry-after": "1911"}, payload=TPD_BODY)
        self._post_returning(monkeypatch, response, slept)
        with pytest.raises(llm.RateLimitExhausted):
            llm._post("https://example.invalid", {"messages": []}, CONFIG)
        assert slept == [], "a daily cap must not be slept on"

    def test_the_providers_own_reason_reaches_the_log(self, monkeypatch):
        slept = []
        response = FakeResponse(headers={"retry-after": "1911"}, payload=TPD_BODY)
        self._post_returning(monkeypatch, response, slept)
        with pytest.raises(llm.RateLimitExhausted, match="tokens per day"):
            llm._post("https://example.invalid", {"messages": []}, CONFIG)

    def test_a_short_wait_is_still_retried(self, monkeypatch):
        slept = []
        response = FakeResponse(headers={"retry-after": "5"}, payload={})
        self._post_returning(monkeypatch, response, slept)
        with pytest.raises(llm.LLMError):
            llm._post("https://example.invalid", {"messages": []}, CONFIG)
        assert slept == [5.0, 5.0, 5.0], "a per-minute limit is worth waiting out"

    def test_exhaustion_is_an_llm_error_too(self):
        """Callers that only catch LLMError must not miss it."""
        assert issubclass(llm.RateLimitExhausted, LLMError)

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

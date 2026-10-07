from types import SimpleNamespace

import pytest
import requests

from app import config, providers
from app.providers import LLMError, OpenAICompatibleLLM, make_llm


def test_openai_compatible_request_and_parse(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(url=url, json=json, headers=headers)
        return SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": " Hello [1]. "}}]})

    monkeypatch.setattr(providers.requests, "post", fake_post)
    llm = OpenAICompatibleLLM("llama3.2", "http://localhost:11434/v1/")
    assert llm.complete("sys", "user", max_tokens=50) == "Hello [1]."
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["json"]["temperature"] == 0
    assert captured["headers"] == {}


def test_connection_error_gives_helpful_message(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(providers.requests, "post", fake_post)
    with pytest.raises(LLMError, match="ollama pull llama3.2"):
        OpenAICompatibleLLM("llama3.2", "http://localhost:11434/v1").complete("s", "u")


def test_default_provider_is_free_ollama(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config, "LLM_MODEL", "")
    monkeypatch.setattr(config, "LLM_BASE_URL", "")
    llm = make_llm()
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.model == "llama3.2"
    assert llm.url == "http://localhost:11434/v1/chat/completions"


def test_unknown_provider(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "nope")
    with pytest.raises(LLMError):
        make_llm()

"""LLM providers behind one small interface: complete(system, user, max_tokens) -> str.

  * ollama            - free, runs open-source models locally (default). https://ollama.com
  * anthropic         - Claude API (paid, needs ANTHROPIC_API_KEY).
  * openai_compatible - any OpenAI-compatible chat endpoint (set LLM_BASE_URL and LLM_API_KEY).
"""
from __future__ import annotations

import requests

from . import config


class LLMError(RuntimeError):
    """Raised when the language model can't be reached or returns an error."""


class AnthropicLLM:
    def __init__(self, model: str):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in message.content if block.type == "text").strip()


class OpenAICompatibleLLM:
    """Works with Ollama's local server and other OpenAI-compatible endpoints."""

    def __init__(self, model: str, base_url: str, api_key: str = ""):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            response = requests.post(self.url, json=payload, headers=self.headers, timeout=300)
        except requests.ConnectionError as exc:
            raise LLMError(
                f"Can't reach the model server at {self.url}. If you're using Ollama, "
                f"make sure it's running and you've pulled the model: ollama pull {self.model}"
            ) from exc
        if response.status_code >= 400:
            raise LLMError(f"Model server error {response.status_code}: {response.text[:300]}")
        return response.json()["choices"][0]["message"]["content"].strip()


def make_llm():
    provider = config.LLM_PROVIDER
    if provider == "anthropic":
        return AnthropicLLM(config.LLM_MODEL or config.ANTHROPIC_MODEL)
    if provider == "ollama":
        return OpenAICompatibleLLM(config.LLM_MODEL or "llama3.2", config.LLM_BASE_URL or "http://localhost:11434/v1")
    if provider == "openai_compatible":
        if not (config.LLM_BASE_URL and config.LLM_MODEL):
            raise LLMError("Set LLM_BASE_URL and LLM_MODEL for the openai_compatible provider.")
        return OpenAICompatibleLLM(config.LLM_MODEL, config.LLM_BASE_URL, config.LLM_API_KEY)
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use ollama, anthropic or openai_compatible.")

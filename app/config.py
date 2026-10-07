"""Settings loaded from environment variables (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Which language model answers questions: "ollama" (free, local), "anthropic" (Claude API)
# or "openai_compatible" (any OpenAI-style endpoint).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")          # e.g. llama3.2 for Ollama; defaults per provider
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")    # Ollama default: http://localhost:11434/v1
LLM_API_KEY = os.getenv("LLM_API_KEY", "")      # only for openai_compatible providers

# Claude model used when LLM_PROVIDER=anthropic.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

# Local embedding model (runs on CPU, no API key needed).
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# The SEC requires a descriptive User-Agent, e.g. "Jane Doe jane@example.com".
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
TOP_K = int(os.getenv("TOP_K", "6"))

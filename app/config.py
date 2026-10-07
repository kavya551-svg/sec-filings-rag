"""Settings loaded from environment variables (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Claude model used for answering and for the groundedness judge.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

# Local embedding model (runs on CPU, no API key needed).
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# The SEC requires a descriptive User-Agent, e.g. "Jane Doe jane@example.com".
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
TOP_K = int(os.getenv("TOP_K", "6"))

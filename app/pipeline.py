"""End-to-end pipeline: ingest a 10-K, then answer questions with citations and evaluation."""
from __future__ import annotations

import json
import time

from . import config, edgar
from .chunking import chunk_text
from .index import Embedder, HybridIndex, SentenceTransformerEmbedder
from .llm import citation_check, generate_answer, is_refusal, judge

_embedder: Embedder | None = None
_client = None
_indexes: dict[str, HybridIndex] = {}


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder(config.EMBED_MODEL)
    return _embedder


def get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _client


def _index_dir(ticker: str):
    return config.DATA_DIR / ticker.upper()


def ingest(ticker: str, embedder: Embedder | None = None) -> dict:
    """Download the latest 10-K for a ticker, chunk it, embed it and save the index."""
    ticker = ticker.upper()
    cik, company = edgar.ticker_to_cik(ticker)
    filing = edgar.latest_10k(cik)
    chunks = chunk_text(edgar.fetch_filing_text(filing["url"]))
    if not chunks:
        raise RuntimeError("The filing produced no text chunks; it may be in an unsupported format.")
    meta = {"ticker": ticker, "company": company, "cik": cik, **filing, "num_chunks": len(chunks)}
    index = HybridIndex.build(chunks, embedder or get_embedder(), meta)
    index.save(_index_dir(ticker))
    _indexes[ticker] = index
    return meta


def get_index(ticker: str) -> HybridIndex:
    ticker = ticker.upper()
    if ticker not in _indexes:
        directory = _index_dir(ticker)
        if not (directory / "chunks.json").exists():
            raise FileNotFoundError(f"{ticker} has not been ingested yet. Ingest it first.")
        _indexes[ticker] = HybridIndex.load(directory, get_embedder())
    return _indexes[ticker]


def list_companies() -> list[dict]:
    if not config.DATA_DIR.exists():
        return []
    companies = []
    for path in sorted(config.DATA_DIR.glob("*/chunks.json")):
        companies.append(json.loads(path.read_text(encoding="utf-8"))["meta"])
    return companies


def ask(ticker: str, question: str, k: int | None = None, evaluate: bool = True, client=None) -> dict:
    """Retrieve, answer with citations, and evaluate the answer."""
    client = client or get_client()
    index = get_index(ticker)
    meta = index.meta

    t0 = time.perf_counter()
    results = index.search(question, k=k or config.TOP_K)
    t1 = time.perf_counter()
    answer = generate_answer(client, question, results, meta.get("company", ticker))
    t2 = time.perf_counter()

    refused = is_refusal(answer)
    evaluation = {"refused": refused, **citation_check(answer, len(results))}
    if evaluate and not refused:
        evaluation.update(judge(client, question, answer, results))
    t3 = time.perf_counter()

    return {
        "question": question,
        "answer": answer,
        "filing": {key: meta.get(key) for key in ("ticker", "company", "form", "filing_date", "url")},
        "sources": [
            {
                "id": i,
                "section": r.chunk.section,
                "text": r.chunk.text,
                "score": round(r.score, 4),
                "dense_rank": r.dense_rank,
                "bm25_rank": r.bm25_rank,
            }
            for i, r in enumerate(results, start=1)
        ],
        "evaluation": evaluation,
        "latency_ms": {
            "retrieval": round((t1 - t0) * 1000),
            "generation": round((t2 - t1) * 1000),
            "evaluation": round((t3 - t2) * 1000),
            "total": round((t3 - t0) * 1000),
        },
    }

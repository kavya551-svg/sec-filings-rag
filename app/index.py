"""Hybrid retrieval: dense embeddings + BM25 keyword search, merged with reciprocal rank fusion.

Dense search finds passages with similar meaning; BM25 catches exact terms that embeddings
often miss in financial text (tickers, product names, figures, defined terms).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalized vector per text."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 64
        )
        return np.asarray(vectors, dtype=np.float32)


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge ranked lists of ids. Each list adds 1 / (k + rank) to a document's score."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    dense_rank: int | None
    bm25_rank: int | None


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder, meta: dict | None = None):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self.meta = meta or {}
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, meta: dict | None = None) -> "HybridIndex":
        return cls(chunks, embedder.encode([c.text for c in chunks]), embedder, meta)

    def search(self, query: str, k: int = 6, candidates: int = 30) -> list[SearchResult]:
        query_vector = self.embedder.encode([query])[0]
        dense_scores = self.embeddings @ query_vector
        dense_ranking = [int(i) for i in np.argsort(-dense_scores)[:candidates]]

        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_ranking = [int(i) for i in np.argsort(-bm25_scores)[:candidates] if bm25_scores[i] > 0]

        dense_pos = {doc_id: rank for rank, doc_id in enumerate(dense_ranking, start=1)}
        bm25_pos = {doc_id: rank for rank, doc_id in enumerate(bm25_ranking, start=1)}
        fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking])[:k]
        return [
            SearchResult(self.chunks[doc_id], score, dense_pos.get(doc_id), bm25_pos.get(doc_id))
            for doc_id, score in fused
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"meta": self.meta, "chunks": [asdict(c) for c in self.chunks]}
        (directory / "chunks.json").write_text(json.dumps(payload), encoding="utf-8")
        np.save(directory / "embeddings.npy", self.embeddings)

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> "HybridIndex":
        payload = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in payload["chunks"]]
        return cls(chunks, np.load(directory / "embeddings.npy"), embedder, payload["meta"])

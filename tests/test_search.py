import hashlib

import numpy as np

from app.chunking import Chunk
from app.index import HybridIndex, reciprocal_rank_fusion, tokenize


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder so tests need no model download."""

    def encode(self, texts):
        vectors = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                vectors[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % 256] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


CHUNKS = [
    Chunk(0, "Item 7. MD&A", "Total net revenue increased 8 percent driven by services growth."),
    Chunk(1, "Item 1C. Cybersecurity", "We maintain a cybersecurity program overseen by the audit committee."),
    Chunk(2, "Item 1. Business", "As of fiscal year end the company had approximately 161,000 full-time employees."),
    Chunk(3, "Item 1A. Risk Factors", "Supply chain disruptions could adversely affect our results."),
]


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    top_two = {doc_id for doc_id, _ in fused[:2]}
    assert top_two == {1, 2}
    assert fused[-1][0] in {3, 4}


def test_hybrid_search_finds_relevant_chunk():
    index = HybridIndex.build(CHUNKS, FakeEmbedder())
    results = index.search("How many employees does the company have?", k=2)
    assert results[0].chunk.id == 2
    assert results[0].bm25_rank == 1


def test_save_and_load_roundtrip(tmp_path):
    embedder = FakeEmbedder()
    index = HybridIndex.build(CHUNKS, embedder, {"ticker": "TEST"})
    index.save(tmp_path)
    loaded = HybridIndex.load(tmp_path, embedder)
    assert loaded.meta == {"ticker": "TEST"}
    assert [c.text for c in loaded.chunks] == [c.text for c in CHUNKS]
    assert loaded.search("cybersecurity program", k=1)[0].chunk.id == 1

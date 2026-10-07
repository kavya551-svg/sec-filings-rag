"""Answer generation with citations, plus automatic evaluation of each answer.

Two checks run on every answer:
  * citation_check: rule-based - which sources were cited, invalid source numbers,
    and how many sentences carry a citation.
  * judge: LLM-as-judge - a second model call scores how well the answer is supported
    by the retrieved sources (groundedness) and lists any unsupported claims.

`client` is any object with complete(system, user, max_tokens) -> str (see providers.py).
"""
from __future__ import annotations

import json
import re

from .index import SearchResult

NO_ANSWER = "The filing excerpts provided don't answer this question."

ANSWER_SYSTEM = f"""You are a financial research assistant answering questions about a company's 10-K filing.

Rules:
- Use ONLY the numbered sources provided. Do not use outside knowledge.
- Cite every factual sentence with source numbers in square brackets, e.g. [2] or [1][3].
- Quote figures exactly as they appear in the sources.
- If the sources do not contain the answer, reply exactly: "{NO_ANSWER}"
- Be concise: a short paragraph or a few bullet points."""

JUDGE_SYSTEM = """You evaluate answers produced by a retrieval-augmented assistant.
Given a question, the numbered sources, and the answer, judge how well every claim in the
answer is supported by the sources. Respond with JSON only, in this exact shape:
{"groundedness": <number from 0 to 1>, "verdict": "grounded" | "partially_grounded" | "not_grounded", "unsupported_claims": [<short strings>]}"""

CITE_RE = re.compile(r"\[(\d+)\]")


def format_sources(results: list[SearchResult]) -> str:
    return "\n\n".join(f"[{i}] ({r.chunk.section})\n{r.chunk.text}" for i, r in enumerate(results, start=1))


def generate_answer(client, question: str, results: list[SearchResult], company: str) -> str:
    return client.complete(
        ANSWER_SYSTEM,
        f"Company: {company}\n\nSources:\n{format_sources(results)}\n\nQuestion: {question}",
        max_tokens=1024,
    )


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in judge response")
    return json.loads(match.group(0))


def judge(client, question: str, answer: str, results: list[SearchResult]) -> dict:
    reply = client.complete(
        JUDGE_SYSTEM,
        f"Question: {question}\n\nSources:\n{format_sources(results)}\n\nAnswer:\n{answer}",
        max_tokens=512,
    )
    try:
        verdict = _parse_json(reply)
        score = float(verdict.get("groundedness"))
        return {
            "groundedness": max(0.0, min(1.0, score)),
            "verdict": str(verdict.get("verdict", "")),
            "unsupported_claims": list(verdict.get("unsupported_claims", [])),
        }
    except (ValueError, TypeError):
        return {"groundedness": None, "verdict": "judge_error", "unsupported_claims": []}


def is_refusal(answer: str) -> bool:
    return NO_ANSWER.lower().rstrip(".") in answer.lower()


def citation_check(answer: str, num_sources: int) -> dict:
    cited = sorted({int(n) for n in CITE_RE.findall(answer)})
    invalid = [n for n in cited if n < 1 or n > num_sources]
    # Move citations placed after the period ("... grew. [1]") in front of it before splitting.
    normalized = re.sub(r"([.!?])\s*((?:\[\d+\])+)", r" \2\1", answer.strip())
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", normalized) if len(s.split()) >= 4]
    with_citation = sum(1 for s in sentences if CITE_RE.search(s))
    return {
        "cited_sources": cited,
        "invalid_citations": invalid,
        "citation_coverage": round(with_citation / len(sentences), 3) if sentences else 0.0,
    }

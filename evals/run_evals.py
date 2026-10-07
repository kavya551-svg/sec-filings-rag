"""Run a fixed question set against an ingested filing and report quality metrics.

    python evals/run_evals.py AAPL

Metrics:
  * Answer/refusal accuracy - answerable questions get an answer; unanswerable ones are refused.
  * Mean groundedness - LLM-judge score for answered questions (0-1).
  * Mean citation coverage - share of answer sentences that carry a citation.
  * Latency p50 / p95.
Results are printed and saved to evals/results_<TICKER>.md.
"""
import argparse
import json
import statistics
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import pipeline  # noqa: E402


def percentile(values: list[float], pct: int) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[pct - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    args = parser.parse_args()
    ticker = args.ticker.upper()

    questions = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))
    rows = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q['question']}")
        result = pipeline.ask(ticker, q["question"])
        ev = result["evaluation"]
        rows.append({
            "question": q["question"],
            "answerable": q["answerable"],
            "refused": ev["refused"],
            "correct_behavior": ev["refused"] != q["answerable"],
            "groundedness": ev.get("groundedness"),
            "citation_coverage": ev["citation_coverage"],
            "latency_ms": result["latency_ms"]["total"],
        })

    answered = [r for r in rows if not r["refused"]]
    grounded = [r["groundedness"] for r in answered if r["groundedness"] is not None]
    latencies = [r["latency_ms"] for r in rows]
    summary = {
        "behavior_accuracy": sum(r["correct_behavior"] for r in rows) / len(rows),
        "mean_groundedness": statistics.mean(grounded) if grounded else None,
        "mean_citation_coverage": statistics.mean(r["citation_coverage"] for r in answered) if answered else None,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
    }

    def fmt(x, pct=True):
        return "n/a" if x is None else (f"{x:.0%}" if pct else f"{x:,.0f}")

    meta = pipeline.get_index(ticker).meta
    lines = [
        f"# Eval results: {meta['company']} ({ticker}) {meta['form']} filed {meta['filing_date']}",
        f"Run on {date.today().isoformat()} with {len(rows)} questions.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Answer/refusal accuracy | {fmt(summary['behavior_accuracy'])} |",
        f"| Mean groundedness | {fmt(summary['mean_groundedness'])} |",
        f"| Mean citation coverage | {fmt(summary['mean_citation_coverage'])} |",
        f"| Latency p50 | {fmt(summary['latency_p50_ms'], pct=False)} ms |",
        f"| Latency p95 | {fmt(summary['latency_p95_ms'], pct=False)} ms |",
        "",
        "| Question | Expected | Got | Groundedness | Citation coverage |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        expected = "answer" if r["answerable"] else "refuse"
        got = "refused" if r["refused"] else "answered"
        lines.append(f"| {r['question']} | {expected} | {got} | {fmt(r['groundedness'])} | {fmt(r['citation_coverage'])} |")

    report = "\n".join(lines) + "\n"
    out = Path(__file__).parent / f"results_{ticker}.md"
    out.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

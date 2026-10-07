"""Command-line interface.

    python cli.py ingest AAPL
    python cli.py ask AAPL "What are the main risk factors?"
"""
import argparse
import textwrap

from app import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="SEC Filings Research Assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_cmd = sub.add_parser("ingest", help="download and index a company's latest 10-K")
    ingest_cmd.add_argument("ticker")

    ask_cmd = sub.add_parser("ask", help="ask a question about an ingested filing")
    ask_cmd.add_argument("ticker")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--k", type=int, default=None, help="number of sources to retrieve")
    ask_cmd.add_argument("--no-eval", action="store_true", help="skip the groundedness judge")

    args = parser.parse_args()

    if args.command == "ingest":
        meta = pipeline.ingest(args.ticker)
        print(f"Indexed {meta['company']} {meta['form']} filed {meta['filing_date']} ({meta['num_chunks']} chunks)")
        print(meta["url"])
        return

    result = pipeline.ask(args.ticker, args.question, k=args.k, evaluate=not args.no_eval)
    ev = result["evaluation"]
    print("\n" + textwrap.fill(result["answer"], width=100, replace_whitespace=False) + "\n")
    if ev.get("groundedness") is not None:
        print(f"Groundedness: {ev['groundedness']:.0%} ({ev['verdict']})")
    print(f"Citation coverage: {ev['citation_coverage']:.0%}   Latency: {result['latency_ms']['total']} ms")
    for claim in ev.get("unsupported_claims", []):
        print(f"  Unsupported: {claim}")
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['id']}] {s['section']}")


if __name__ == "__main__":
    main()

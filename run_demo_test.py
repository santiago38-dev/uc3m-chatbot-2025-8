"""
Batch Demo Test Runner for RAG System

Runs multiple queries through both Flash and Thinking modes,
records responses, timing, and sources, then saves results to JSON.

Usage:
    python run_demo_test.py                          # Use default demo_queries.json
    python run_demo_test.py --input queries.json    # Custom JSON input
    python run_demo_test.py --input queries.txt     # Plain text (one query per line)
    python run_demo_test.py --mode flash            # Only run Flash mode
    python run_demo_test.py --mode thinking         # Only run Thinking mode
    python run_demo_test.py --output results.json   # Custom output file
"""

import argparse
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from src.rag_advanced import (
    get_flash_chain,
    get_thinking_chain,
    set_verbose,
    config
)
from src.vector_store import get_smart_retriever

# Configuration
K_DOCS = config.K_DOCS_DEFAULT
DEFAULT_INPUT = "demo_queries.json"
DEFAULT_OUTPUT = "demo_test_results.json"


def load_queries(input_path: str) -> List[Dict[str, str]]:
    """Load queries from JSON or TXT file."""
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Support both {"queries": [...]} and direct list format
            if isinstance(data, dict) and "queries" in data:
                return data["queries"]
            elif isinstance(data, list):
                return [{"id": f"Q{i+1}", "query": q} if isinstance(q, str) else q
                        for i, q in enumerate(data)]
            else:
                raise ValueError("JSON must contain 'queries' array or be a list")

    elif path.suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            return [{"id": f"Q{i+1}", "query": q} for i, q in enumerate(lines)]

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def extract_sources_from_response(response: str) -> List[str]:
    """Extract source citations from response text."""
    sources = []

    # Pattern 1: [Source N: Project (INR) - Section]
    pattern1 = re.findall(r'\[Source \d+: ([^\]]+)\]', response)
    sources.extend(pattern1)

    # Pattern 2: Sources section at end
    sources_section = re.search(r'Sources?:\s*\n((?:\s*\[\d+\][^\n]+\n?)+)', response, re.IGNORECASE)
    if sources_section:
        citations = re.findall(r'\[\d+\]\s*([^\n]+)', sources_section.group(1))
        sources.extend(citations)

    # Pattern 3: Inline references like [1], [2], etc.
    inline_refs = re.findall(r'\[(\d+)\]', response)

    return {
        "extracted_sources": list(set(sources)),
        "inline_references": list(set(inline_refs))
    }


def run_query(chain, query: str, session_id: str) -> Dict[str, Any]:
    """Run a single query and collect the full response."""
    start_time = time.time()
    response_chunks = []

    for chunk in chain.stream(
        {"question": query},
        config={"configurable": {"session_id": session_id}}
    ):
        response_chunks.append(chunk)

    elapsed_time = time.time() - start_time
    full_response = "".join(response_chunks)
    sources = extract_sources_from_response(full_response)

    return {
        "response": full_response,
        "response_time_seconds": round(elapsed_time, 2),
        "sources": sources
    }


def run_batch_test(
    queries: List[Dict[str, str]],
    modes: List[str] = ["flash", "thinking"],
    verbose: bool = True
) -> Dict[str, Any]:
    """Run batch test across specified modes."""

    print("=" * 70)
    print("BATCH DEMO TEST RUNNER")
    print("=" * 70)
    print(f"Queries to run: {len(queries)}")
    print(f"Modes: {', '.join(modes)}")
    print("=" * 70)

    # Initialize retriever once (shared across both modes)
    print("\nLoading retriever...")
    retriever = get_smart_retriever(k_docs=K_DOCS)
    print("Retriever ready.\n")

    # Disable verbose logging for cleaner output
    set_verbose(enabled=False)

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(queries),
            "modes_tested": modes
        },
        "results": []
    }

    total_tests = len(queries) * len(modes)
    current_test = 0

    for query_item in queries:
        query_id = query_item.get("id", "?")
        query_text = query_item.get("query", query_item.get("text", ""))

        query_result = {
            "id": query_id,
            "query": query_text,
            "modes": {}
        }

        for mode in modes:
            current_test += 1
            print(f"[{current_test}/{total_tests}] {query_id} ({mode.upper()})...")

            # Get appropriate chain
            if mode == "flash":
                chain = get_flash_chain(retriever, with_history=False)
            else:
                chain = get_thinking_chain(retriever, with_history=False)

            # Create unique session ID
            session_id = f"batch_{query_id}_{mode}_{int(time.time())}"

            try:
                result = run_query(chain, query_text, session_id)
                query_result["modes"][mode] = {
                    "status": "success",
                    **result
                }
                print(f"    Done in {result['response_time_seconds']}s")

            except Exception as e:
                query_result["modes"][mode] = {
                    "status": "error",
                    "error": str(e),
                    "response_time_seconds": 0
                }
                print(f"    ERROR: {str(e)[:50]}")

        results["results"].append(query_result)

    # Calculate summary statistics
    flash_times = [r["modes"].get("flash", {}).get("response_time_seconds", 0)
                   for r in results["results"] if "flash" in r["modes"]]
    thinking_times = [r["modes"].get("thinking", {}).get("response_time_seconds", 0)
                      for r in results["results"] if "thinking" in r["modes"]]

    results["summary"] = {}

    if flash_times:
        results["summary"]["flash"] = {
            "avg_time": round(sum(flash_times) / len(flash_times), 2),
            "min_time": round(min(flash_times), 2),
            "max_time": round(max(flash_times), 2),
            "total_time": round(sum(flash_times), 2)
        }

    if thinking_times:
        results["summary"]["thinking"] = {
            "avg_time": round(sum(thinking_times) / len(thinking_times), 2),
            "min_time": round(min(thinking_times), 2),
            "max_time": round(max(thinking_times), 2),
            "total_time": round(sum(thinking_times), 2)
        }

    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


def print_summary(results: Dict[str, Any]):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = results.get("summary", {})

    for mode, stats in summary.items():
        print(f"\n{mode.upper()} Mode:")
        print(f"  Average response time: {stats['avg_time']}s")
        print(f"  Min/Max: {stats['min_time']}s / {stats['max_time']}s")
        print(f"  Total time: {stats['total_time']}s")

    if "flash" in summary and "thinking" in summary:
        overhead = summary["thinking"]["avg_time"] - summary["flash"]["avg_time"]
        ratio = summary["thinking"]["avg_time"] / summary["flash"]["avg_time"] if summary["flash"]["avg_time"] > 0 else 0
        print(f"\nThinking overhead: +{overhead:.2f}s ({ratio:.1f}x slower)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run batch demo tests on RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Input file with queries (JSON or TXT). Default: {DEFAULT_INPUT}"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output file for results. Default: {DEFAULT_OUTPUT}"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["flash", "thinking", "both"],
        default="both",
        help="Which mode(s) to test. Default: both"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of queries to run (for quick testing)"
    )

    args = parser.parse_args()

    # Determine modes to run
    if args.mode == "both":
        modes = ["flash", "thinking"]
    else:
        modes = [args.mode]

    # Load queries
    queries = load_queries(args.input)

    # Apply limit if specified
    if args.limit:
        queries = queries[:args.limit]

    # Run tests
    results = run_batch_test(queries, modes=modes)

    # Save and summarize
    save_results(results, args.output)
    print_summary(results)

    print("\nDone!")


if __name__ == "__main__":
    main()

"""
ERCOT RAG Demo Questions - Test Flash vs Thinking Modes
========================================================

6 demo questions to test the RAG chatbot with both modes.

Run: python demo_questions.py
     python demo_questions.py --mode flash
     python demo_questions.py --mode thinking
     python demo_questions.py --output results.json
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# Verify we're in the right directory
if not Path("src/rag_advanced").exists():
    print("ERROR: Must run from the uc3m-chatbot-2025-8 directory")
    sys.exit(1)

from src.rag_advanced import (
    get_flash_chain,
    get_thinking_chain,
    get_session_history,
    set_verbose,
)
from src.vector_store import get_smart_retriever, CHROMADB_PATH

K_DOCS = 15

# Demo questions - Finance, Legal, Competitive Intelligence
DEMO_QUESTIONS = [
    {
        "id": 1,
        "category": "Finance",
        "title": "Battery vs Solar Security Costs",
        "question": "What's the range of security deposits per kW for battery storage projects? How does this compare to solar projects?",
    },
    {
        "id": 2,
        "category": "Finance",
        "title": "High Security Outliers",
        "question": "Which projects have security amounts above $100 per kW? What might explain these high costs?",
    },
    {
        "id": 3,
        "category": "Legal",
        "title": "Force Majeure Comparison",
        "question": "Compare the force majeure provisions in Centerpoint vs ONCOR interconnection agreements. What are the key differences?",
    },
    {
        "id": 4,
        "category": "Legal",
        "title": "Payment Default Cure Periods",
        "question": "What cure periods are specified for payment defaults in ERCOT interconnection agreements?",
    },
    {
        "id": 5,
        "category": "Competitive",
        "title": "Developer Comparison",
        "question": "Compare RWE and NextEra solar projects. What differences do you see in their interconnection terms?",
    },
    {
        "id": 6,
        "category": "Competitive",
        "title": "West Texas Projects",
        "question": "Show me battery storage projects in West Texas. What are their security requirements and TSP assignments?",
    },
]


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    line = char * 70
    print(f"\n{line}")
    print(f"  {text}")
    print(line)


def run_question(chain, question_data: dict, session_id: str) -> tuple:
    """
    Run a single question through the chain.
    Returns (response_text, elapsed_time, error).
    """
    q = question_data["question"]

    print(f"\nQ{question_data['id']}: {question_data['title']}")
    print(f"Category: {question_data['category']}")
    print(f"\nQuestion: {q}")
    print("\n" + "-" * 50)
    print("Response:\n")

    start = time.time()
    full_response = ""
    error = None

    try:
        for chunk in chain.stream(
            {"question": q},
            config={"configurable": {"session_id": session_id}}
        ):
            print(chunk, end="", flush=True)
            full_response += str(chunk)
    except Exception as e:
        error = str(e)
        print(f"\n\nERROR: {error}")

    elapsed = time.time() - start
    print(f"\n\n[Time: {elapsed:.2f}s]")

    # Clear session history to avoid cross-contamination between questions
    try:
        get_session_history(session_id).clear()
    except Exception:
        pass

    return full_response, elapsed, error


def verify_chromadb():
    """Check if ChromaDB exists and is accessible."""
    db_path = Path(CHROMADB_PATH)
    if not db_path.exists():
        print(f"ERROR: ChromaDB not found at: {db_path.absolute()}")
        print(f"\nExpected path: {CHROMADB_PATH}")
        print("\nMake sure you have:")
        print("  1. The ercot-lgia-rag-system repo as a sibling directory")
        print("  2. Run the indexing pipeline to create the ChromaDB")
        print("  3. Or set CHROMADB_PATH environment variable")
        return False
    return True


def run_demo(mode: str = "both", pause: bool = False, verbose: bool = False, output_file: str = None):
    """Run the demo with specified mode(s)."""

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  ERCOT RAG DEMO - Testing Flash vs Thinking Modes  ".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    # Verify ChromaDB exists
    if not verify_chromadb():
        sys.exit(1)

    # Load retriever
    print("\nLoading retriever...")
    try:
        retriever = get_smart_retriever(k_docs=K_DOCS)
        print(f"SmartRetriever loaded (k={K_DOCS})")
        print(f"ChromaDB path: {CHROMADB_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to load retriever: {e}")
        sys.exit(1)

    # Set verbose mode
    set_verbose(enabled=verbose)

    # Generate unique run ID for session isolation
    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().isoformat()

    results = {
        "run_id": run_id,
        "timestamp": timestamp,
        "mode": mode,
        "k_docs": K_DOCS,
        "chromadb_path": CHROMADB_PATH,
        "flash": [],
        "thinking": []
    }

    try:
        # Run Flash mode
        if mode in ("flash", "both"):
            print_header("FLASH MODE (Fast)", "=")
            flash_chain = get_flash_chain(retriever)

            for i, q_data in enumerate(DEMO_QUESTIONS):
                session_id = f"demo_{run_id}_flash_{i}"
                response, elapsed, error = run_question(flash_chain, q_data, session_id)
                results["flash"].append({
                    "id": q_data["id"],
                    "title": q_data["title"],
                    "category": q_data["category"],
                    "question": q_data["question"],
                    "response": response,
                    "time": elapsed,
                    "response_length": len(response),
                    "error": error
                })

                if pause and i < len(DEMO_QUESTIONS) - 1:
                    input("\n[Press Enter for next question...]")

        # Run Thinking mode
        if mode in ("thinking", "both"):
            print_header("THINKING MODE (Deep Verification)", "=")
            thinking_chain = get_thinking_chain(retriever)

            for i, q_data in enumerate(DEMO_QUESTIONS):
                session_id = f"demo_{run_id}_thinking_{i}"
                response, elapsed, error = run_question(thinking_chain, q_data, session_id)
                results["thinking"].append({
                    "id": q_data["id"],
                    "title": q_data["title"],
                    "category": q_data["category"],
                    "question": q_data["question"],
                    "response": response,
                    "time": elapsed,
                    "response_length": len(response),
                    "error": error
                })

                if pause and i < len(DEMO_QUESTIONS) - 1:
                    input("\n[Press Enter for next question...]")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        results["interrupted"] = True

    # Summary
    print_header("PERFORMANCE SUMMARY", "=")

    if results["flash"]:
        flash_times = [r["time"] for r in results["flash"]]
        flash_errors = sum(1 for r in results["flash"] if r["error"])
        print(f"\nFlash Mode ({len(results['flash'])} questions, {flash_errors} errors):")
        print(f"  Total time:   {sum(flash_times):.2f}s")
        print(f"  Avg per Q:    {sum(flash_times)/len(flash_times):.2f}s")
        print(f"  Fastest:      {min(flash_times):.2f}s")
        print(f"  Slowest:      {max(flash_times):.2f}s")

    if results["thinking"]:
        thinking_times = [r["time"] for r in results["thinking"]]
        thinking_errors = sum(1 for r in results["thinking"] if r["error"])
        print(f"\nThinking Mode ({len(results['thinking'])} questions, {thinking_errors} errors):")
        print(f"  Total time:   {sum(thinking_times):.2f}s")
        print(f"  Avg per Q:    {sum(thinking_times)/len(thinking_times):.2f}s")
        print(f"  Fastest:      {min(thinking_times):.2f}s")
        print(f"  Slowest:      {max(thinking_times):.2f}s")

    if results["flash"] and results["thinking"]:
        flash_total = sum(r["time"] for r in results["flash"])
        thinking_total = sum(r["time"] for r in results["thinking"])
        if flash_total > 0:
            overhead = thinking_total - flash_total
            pct = (thinking_total / flash_total - 1) * 100
            print(f"\nOverhead (Thinking vs Flash): +{overhead:.2f}s ({pct:.0f}% slower)")

    # Save results to file if requested
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"\nWARNING: Failed to save results: {e}")

    print("\n" + "=" * 70)
    print(f"DEMO COMPLETE (Run ID: {run_id})")
    print("=" * 70 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ERCOT RAG demo questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo_questions.py                    # Run both modes
    python demo_questions.py --mode flash       # Flash mode only
    python demo_questions.py --mode thinking    # Thinking mode only
    python demo_questions.py --list             # List questions only
    python demo_questions.py --output results.json  # Save results to file
        """
    )
    parser.add_argument(
        "--mode",
        choices=["flash", "thinking", "both"],
        default="both",
        help="Which mode(s) to run (default: both)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Just list the demo questions without running"
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Pause between questions (press Enter to continue)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose RAG processing logs"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        metavar="FILE",
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    if args.list:
        print("\nDemo Questions:")
        print("-" * 60)
        for q in DEMO_QUESTIONS:
            print(f"\nQ{q['id']}: {q['title']} [{q['category']}]")
            print(f"   {q['question']}")
        print("\n" + "-" * 60)
        print(f"Total: {len(DEMO_QUESTIONS)} questions")
        print()
        return

    run_demo(
        mode=args.mode,
        pause=args.pause,
        verbose=args.verbose,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

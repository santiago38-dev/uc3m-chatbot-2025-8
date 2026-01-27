"""
ERCOT RAG Demo Questions - Test Flash vs Thinking Modes
========================================================

6 demo questions to test the RAG chatbot with both modes.

Run: python demo_questions.py
     python demo_questions.py --mode flash
     python demo_questions.py --mode thinking
"""

import argparse
import time
from src.rag_advanced import (
    get_flash_chain,
    get_thinking_chain,
    set_verbose,
)
from src.vector_store import get_smart_retriever

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
    Returns (response_text, elapsed_time).
    """
    q = question_data["question"]

    print(f"\nQ{question_data['id']}: {question_data['title']}")
    print(f"Category: {question_data['category']}")
    print(f"\nQuestion: {q}")
    print("\n" + "-" * 50)
    print("Response:\n")

    start = time.time()
    full_response = ""

    for chunk in chain.stream(
        {"question": q},
        config={"configurable": {"session_id": session_id}}
    ):
        print(chunk, end="", flush=True)
        full_response += str(chunk)

    elapsed = time.time() - start
    print(f"\n\n[Time: {elapsed:.2f}s]")

    return full_response, elapsed


def run_demo(mode: str = "both", pause: bool = False):
    """Run the demo with specified mode(s)."""

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  ERCOT RAG DEMO - Testing Flash vs Thinking Modes  ".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    # Load retriever
    print("\nLoading retriever...")
    retriever = get_smart_retriever(k_docs=K_DOCS)
    print(f"SmartRetriever loaded (k={K_DOCS})")

    # Disable verbose logging for cleaner output
    set_verbose(enabled=False)

    results = {"flash": [], "thinking": []}

    # Run Flash mode
    if mode in ("flash", "both"):
        print_header("FLASH MODE (Fast)", "=")
        flash_chain = get_flash_chain(retriever)

        for i, q_data in enumerate(DEMO_QUESTIONS):
            session_id = f"demo_flash_{i}"
            response, elapsed = run_question(flash_chain, q_data, session_id)
            results["flash"].append({
                "id": q_data["id"],
                "title": q_data["title"],
                "time": elapsed,
                "response_length": len(response)
            })

            if pause and i < len(DEMO_QUESTIONS) - 1:
                input("\n[Press Enter for next question...]")

    # Run Thinking mode
    if mode in ("thinking", "both"):
        print_header("THINKING MODE (Deep Verification)", "=")
        thinking_chain = get_thinking_chain(retriever)

        for i, q_data in enumerate(DEMO_QUESTIONS):
            session_id = f"demo_thinking_{i}"
            response, elapsed = run_question(thinking_chain, q_data, session_id)
            results["thinking"].append({
                "id": q_data["id"],
                "title": q_data["title"],
                "time": elapsed,
                "response_length": len(response)
            })

            if pause and i < len(DEMO_QUESTIONS) - 1:
                input("\n[Press Enter for next question...]")

    # Summary
    print_header("PERFORMANCE SUMMARY", "=")

    if results["flash"]:
        flash_times = [r["time"] for r in results["flash"]]
        print(f"\nFlash Mode:")
        print(f"  Total time:   {sum(flash_times):.2f}s")
        print(f"  Avg per Q:    {sum(flash_times)/len(flash_times):.2f}s")
        print(f"  Fastest:      {min(flash_times):.2f}s")
        print(f"  Slowest:      {max(flash_times):.2f}s")

    if results["thinking"]:
        thinking_times = [r["time"] for r in results["thinking"]]
        print(f"\nThinking Mode:")
        print(f"  Total time:   {sum(thinking_times):.2f}s")
        print(f"  Avg per Q:    {sum(thinking_times)/len(thinking_times):.2f}s")
        print(f"  Fastest:      {min(thinking_times):.2f}s")
        print(f"  Slowest:      {max(thinking_times):.2f}s")

    if results["flash"] and results["thinking"]:
        flash_total = sum(r["time"] for r in results["flash"])
        thinking_total = sum(r["time"] for r in results["thinking"])
        overhead = thinking_total - flash_total
        print(f"\nOverhead (Thinking vs Flash): +{overhead:.2f}s ({(thinking_total/flash_total - 1)*100:.0f}% slower)")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run ERCOT RAG demo questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo_questions.py                # Run both modes
    python demo_questions.py --mode flash   # Flash mode only
    python demo_questions.py --mode thinking # Thinking mode only
    python demo_questions.py --list         # List questions only
        """
    )
    parser.add_argument(
        "--mode",
        choices=["flash", "thinking", "both"],
        default="both",
        help="Which mode(s) to run"
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

    args = parser.parse_args()

    if args.list:
        print("\nDemo Questions:")
        print("-" * 50)
        for q in DEMO_QUESTIONS:
            print(f"\nQ{q['id']}: {q['title']} [{q['category']}]")
            print(f"   {q['question']}")
        print()
        return

    run_demo(mode=args.mode, pause=args.pause)


if __name__ == "__main__":
    main()

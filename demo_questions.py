#!/usr/bin/env python3
"""
Demo Questions - Test ERCOT RAG comparative query fixes.

Usage:
    python demo_questions.py --mode flash --verbose
    python demo_questions.py --mode thinking
    python demo_questions.py --question "Compare RWE vs SAMSUNG"
"""

import argparse
import sys
from collections import Counter

# Test questions from the issue
TEST_QUESTIONS = {
    "Q1": "Compare battery vs solar project security deposits",
    "Q2": "Which projects have security deposits over $100/kW?",
    "Q3": "Compare ONCOR vs Centerpoint interconnection requirements",
    "Q4": "What are the typical cure periods in ERCOT agreements?",
    "Q5": "Compare RWE vs SAMSUNG battery storage projects",
    "Q6": "List all battery storage projects in the West Texas zone",
}


def test_filter_extraction():
    """Test that filter extraction works correctly."""
    print("=" * 70)
    print("TESTING FILTER EXTRACTION")
    print("=" * 70)

    from src.vector_store import extract_multi_filters_from_query, build_chromadb_where_clause

    for qid, question in TEST_QUESTIONS.items():
        print(f"\n{qid}: {question}")
        filters = extract_multi_filters_from_query(question)
        print(f"  Filters: {filters}")

        if filters:
            where = build_chromadb_where_clause(filters, expand_aliases=True)
            print(f"  Where clause keys: {list(where.keys()) if where else 'None'}")

            # Show expanded values for parent_company
            if 'parent_company' in filters:
                from src.rag_advanced.alias_expander import expand_parent_company_aliases
                if isinstance(filters['parent_company'], list):
                    expanded = expand_parent_company_aliases(filters['parent_company'])
                    print(f"  Expanded parent_company: {expanded}")


def test_retrieval(question: str, k: int = 15, verbose: bool = False):
    """Test document retrieval for a question."""
    print("\n" + "=" * 70)
    print(f"TESTING RETRIEVAL: {question}")
    print("=" * 70)

    from src.vector_store import (
        get_smart_retriever,
        extract_multi_filters_from_query,
        build_chromadb_where_clause
    )

    # Extract filters
    filters = extract_multi_filters_from_query(question)
    print(f"\nExtracted filters: {filters}")

    where_clause = build_chromadb_where_clause(filters, expand_aliases=True)
    print(f"Where clause: {where_clause}")

    # Get retriever
    retriever = get_smart_retriever(k_docs=k)

    # Search with hard filters
    if where_clause:
        print(f"\nUsing HARD filtering...")
        docs = retriever.search_with_hard_filters(question, where=where_clause, k=k)
    else:
        print(f"\nUsing standard retrieval...")
        docs = retriever.invoke(question)

    print(f"\nRetrieved {len(docs)} documents")

    # Analyze results
    if docs:
        # Developer distribution
        devs = Counter(d.metadata.get('parent_company', 'unknown') for d in docs)
        print(f"\nDeveloper distribution:")
        for dev, count in devs.most_common():
            print(f"  {dev}: {count} chunks")

        # Project distribution
        projects = Counter(d.metadata.get('project_name', 'unknown') for d in docs)
        print(f"\nProject distribution:")
        for proj, count in projects.most_common(10):
            print(f"  {proj}: {count} chunks")

        # Fuel type distribution
        fuels = Counter(d.metadata.get('fuel_type', 'unknown') for d in docs)
        print(f"\nFuel type distribution:")
        for fuel, count in fuels.most_common():
            print(f"  {fuel}: {count} chunks")

        if verbose:
            print(f"\nDocument details:")
            for i, doc in enumerate(docs[:5]):
                meta = doc.metadata
                print(f"  [{i+1}] {meta.get('project_name', 'N/A')} | "
                      f"{meta.get('parent_company', 'N/A')} | "
                      f"{meta.get('inr', 'N/A')} | "
                      f"{meta.get('section', 'N/A')}")

    return docs


def test_full_chain(question: str, mode: str = "flash", verbose: bool = False):
    """Test the full RAG chain."""
    print("\n" + "=" * 70)
    print(f"TESTING FULL CHAIN ({mode.upper()}): {question}")
    print("=" * 70)

    from src.vector_store import get_smart_retriever
    from src.rag_advanced.chain import get_rag_chain
    from src.rag_advanced.utils import RAGMode

    # Get retriever and chain
    retriever = get_smart_retriever(k_docs=15)
    rag_mode = RAGMode.FLASH if mode == "flash" else RAGMode.THINKING
    chain = get_rag_chain(retriever, mode=rag_mode, with_history=False)

    # Run chain
    print("\nGenerating response...")
    response_parts = []
    for chunk in chain.stream({"question": question}):
        response_parts.append(chunk)
        if verbose:
            print(chunk, end="", flush=True)

    response = "".join(response_parts)

    if not verbose:
        print(f"\nResponse preview (first 500 chars):")
        print(response[:500])
        if len(response) > 500:
            print("...")

    return response


def test_all_questions(mode: str = "flash", verbose: bool = False):
    """Test all questions with specified mode."""
    print("\n" + "=" * 70)
    print(f"TESTING ALL QUESTIONS - {mode.upper()} MODE")
    print("=" * 70)

    results = {}
    for qid, question in TEST_QUESTIONS.items():
        print(f"\n{'='*70}")
        print(f"{qid}: {question}")
        print("=" * 70)

        try:
            response = test_full_chain(question, mode=mode, verbose=verbose)
            results[qid] = {"status": "OK", "response_len": len(response)}
        except Exception as e:
            results[qid] = {"status": "ERROR", "error": str(e)}
            print(f"ERROR: {e}")

        print("\n" + "-" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for qid, result in results.items():
        status = "✅" if result["status"] == "OK" else "❌"
        print(f"  {qid}: {status} {result['status']}")

    return results


def test_all_modes(verbose: bool = False):
    """Test all questions with BOTH flash and thinking modes."""
    print("\n" + "=" * 70)
    print("TESTING ALL QUESTIONS - BOTH MODES")
    print("=" * 70)

    all_results = {}

    for mode in ["flash", "thinking"]:
        print(f"\n\n{'#'*70}")
        print(f"# {mode.upper()} MODE")
        print(f"{'#'*70}")
        all_results[mode] = test_all_questions(mode=mode, verbose=verbose)

    # Final comparison
    print("\n\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Question':<10} {'Flash':<15} {'Thinking':<15}")
    print("-" * 40)
    for qid in TEST_QUESTIONS.keys():
        flash_status = "✅" if all_results["flash"].get(qid, {}).get("status") == "OK" else "❌"
        think_status = "✅" if all_results["thinking"].get(qid, {}).get("status") == "OK" else "❌"
        print(f"{qid:<10} {flash_status:<15} {think_status:<15}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Test ERCOT RAG comparative queries")
    parser.add_argument("--mode", choices=["flash", "thinking", "both"], default="flash",
                        help="RAG mode to use (both = test flash AND thinking)")
    parser.add_argument("--question", "-q", type=str,
                        help="Custom question to test")
    parser.add_argument("--test", "-t", choices=["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"],
                        help="Run specific test question")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Run ALL test questions")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--filters-only", action="store_true",
                        help="Only test filter extraction (no LLM calls)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Only test retrieval (no LLM generation)")

    args = parser.parse_args()

    # Filter extraction test
    if args.filters_only:
        test_filter_extraction()
        return

    # Run all questions
    if args.all:
        if args.mode == "both":
            test_all_modes(verbose=args.verbose)
        else:
            test_all_questions(mode=args.mode, verbose=args.verbose)
        return

    # Determine question to test
    if args.question:
        question = args.question
    elif args.test:
        question = TEST_QUESTIONS[args.test]
    else:
        # Default to Q5 (the problematic one)
        question = TEST_QUESTIONS["Q5"]

    # Retrieval-only test
    if args.retrieval_only:
        test_retrieval(question, verbose=args.verbose)
        return

    # Full chain test - support "both" mode for single question too
    if args.mode == "both":
        print("Testing with FLASH mode:")
        test_full_chain(question, mode="flash", verbose=args.verbose)
        print("\n\nTesting with THINKING mode:")
        test_full_chain(question, mode="thinking", verbose=args.verbose)
    else:
        test_full_chain(question, mode=args.mode, verbose=args.verbose)


if __name__ == "__main__":
    main()

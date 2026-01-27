#!/usr/bin/env python3
"""
Demo Questions - Production-grade ERCOT RAG test suite.

Includes:
- Original Q1-Q6 comparative query tests
- C-Suite demo questions (CEO, CFO, Legal, Development, Tech)
- Analytics routing tests
- Role-based filtering

Usage:
    python demo_questions.py --mode flash --verbose
    python demo_questions.py --mode thinking
    python demo_questions.py --question "Compare RWE vs SAMSUNG"
    python demo_questions.py --all --mode both
    python demo_questions.py --role ceo                    # CEO questions only
    python demo_questions.py --role cfo                    # CFO questions only
    python demo_questions.py --analytics-only              # Analytics questions only
    python demo_questions.py -t Q7                         # Specific question
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# QUESTION DEFINITIONS - Organized by Role
# =============================================================================

# Question metadata: (question_text, role, query_type, notes)
# Query types: "comparative", "analytics", "hard_filter", "semantic", "single_lookup"

TEST_QUESTIONS_METADATA = {
    # --- ORIGINAL Q1-Q6: Core Comparative Query Tests ---
    "Q1": {
        "question": "Compare battery vs solar project security deposits",
        "role": "Core",
        "type": "comparative",
        "notes": "Fuel type comparison - tests multi-value filter"
    },
    "Q2": {
        "question": "Which projects have security deposits over $100/kW?",
        "role": "Core",
        "type": "hard_filter",
        "notes": "Numeric threshold filter"
    },
    "Q3": {
        "question": "Compare ONCOR vs Centerpoint interconnection requirements",
        "role": "Core",
        "type": "comparative",
        "notes": "TSP comparison - tests entity-type-aware warnings"
    },
    "Q4": {
        "question": "What are the typical cure periods in ERCOT agreements?",
        "role": "Core",
        "type": "semantic",
        "notes": "Pattern extraction - tests deduplication fix"
    },
    "Q5": {
        "question": "Compare RWE vs SAMSUNG battery storage projects",
        "role": "Core",
        "type": "comparative",
        "notes": "Developer comparison - tests alias expansion + attribution"
    },
    "Q6": {
        "question": "List all battery storage projects in the West Texas zone",
        "role": "Core",
        "type": "hard_filter",
        "notes": "Zone + fuel_type filter - tests deduplication"
    },

    # --- CEO / CHIEF STRATEGY OFFICER ---
    "Q7": {
        "question": "Which developers have the most diversified portfolios across battery, solar, and wind?",
        "role": "CEO",
        "type": "analytics",
        "notes": "M&A screening - uses developer_analysis.diversified_portfolios"
    },
    "Q8": {
        "question": "Compare RWE's security deposits to the corpus median - are they paying above or below market?",
        "role": "CEO",
        "type": "analytics",
        "notes": "Competitor benchmarking - uses specific_developers + corpus_stats"
    },
    "Q9": {
        "question": "What geographic concentration patterns exist? Which counties have the most projects?",
        "role": "CEO",
        "type": "analytics",
        "notes": "Market saturation - uses geographic_concentration"
    },

    # --- CFO / HEAD OF FINANCE ---
    "Q10": {
        "question": "What is the range of security deposits per kW for battery storage? How does this compare to solar?",
        "role": "CFO",
        "type": "analytics",
        "notes": "THE finance question - uses by_fuel_type stats"
    },
    "Q11": {
        "question": "Rank the TSPs by average security requirement per kW. Which is most expensive?",
        "role": "CFO",
        "type": "analytics",
        "notes": "TSP cost benchmarking - uses tsp_rankings"
    },

    # --- HEAD OF DEVELOPMENT / VP ORIGINATION ---
    "Q12": {
        "question": "Which developers have projects in multiple ERCOT zones?",
        "role": "Development",
        "type": "analytics",
        "notes": "Geographic diversification - uses developer_analysis.multi_zone_developers"
    },
    "Q13": {
        "question": "Compare Headcamp Energy Storage Plant to Quantum Storage - what are the key differences?",
        "role": "Development",
        "type": "comparative",
        "notes": "Project-to-project comparison for deal benchmarking"
    },

    # --- GENERAL COUNSEL / CHIEF LEGAL OFFICER ---
    "Q14": {
        "question": "Compare force majeure provisions in Centerpoint vs ONCOR agreements",
        "role": "Legal",
        "type": "comparative",
        "notes": "Contract risk analysis - TSP filter + semantic"
    },
    "Q15": {
        "question": "What termination rights does the Transmission Service Provider reserve in these agreements?",
        "role": "Legal",
        "type": "semantic",
        "notes": "The kill switch question - when can TSP strand investment"
    },
    "Q16": {
        "question": "What are the liability limitations for ERCOT in interconnection agreements?",
        "role": "Legal",
        "type": "semantic",
        "notes": "Risk allocation with grid operator"
    },

    # --- HEAD OF TECHNOLOGY / CIO ---
    "Q17": {
        "question": "What is the security deposit for Quantum Storage?",
        "role": "Tech",
        "type": "single_lookup",
        "notes": "Precision demo - needle in haystack exact match"
    },
    "Q18": {
        "question": "Show me all the details for project INR 25INR0138",
        "role": "Tech",
        "type": "single_lookup",
        "notes": "INR lookup - tests exact match retrieval"
    },
}

# Simple dict for backward compatibility
TEST_QUESTIONS = {qid: meta["question"] for qid, meta in TEST_QUESTIONS_METADATA.items()}

# Role mappings
ROLES = {
    "core": "Core",
    "ceo": "CEO",
    "cfo": "CFO",
    "development": "Development",
    "legal": "Legal",
    "tech": "Tech",
}


# =============================================================================
# ANALYTICS HELPER
# =============================================================================

def load_analytics() -> Optional[Dict]:
    """Load corpus analytics for analytics-type questions."""
    analytics_path = Path("data/corpus_analytics.json")
    if analytics_path.exists():
        with open(analytics_path) as f:
            return json.load(f)
    return None


def test_analytics_availability():
    """Test which analytics fields are available."""
    print("=" * 70)
    print("ANALYTICS DATA AVAILABILITY")
    print("=" * 70)

    analytics = load_analytics()
    if not analytics:
        print("❌ Analytics file not found at data/corpus_analytics.json")
        return

    # Check key fields
    fields_to_check = [
        ("corpus_stats", "Basic corpus statistics"),
        ("corpus_stats.total_projects", "Total project count"),
        ("corpus_stats.security_per_kw.median", "Median $/kW"),
        ("by_fuel_type", "Stats by fuel type"),
        ("by_zone", "Stats by zone"),
        ("tsp_rankings", "TSP rankings"),
        ("developer_analysis", "Developer analysis"),
        ("developer_analysis.multi_zone_developers", "Multi-zone developers"),
        ("developer_analysis.diversified_portfolios", "Diversified portfolios"),
        ("geographic_concentration", "Geographic concentration"),
        ("specific_developers", "Specific developer stats"),
    ]

    for field_path, description in fields_to_check:
        parts = field_path.split(".")
        data = analytics
        found = True
        for part in parts:
            if isinstance(data, dict) and part in data:
                data = data[part]
            else:
                found = False
                break

        status = "✅" if found else "❌"
        value_preview = ""
        if found:
            if isinstance(data, (int, float)):
                value_preview = f" = {data}"
            elif isinstance(data, list):
                value_preview = f" ({len(data)} items)"
            elif isinstance(data, dict):
                value_preview = f" ({len(data)} keys)"
        print(f"  {status} {field_path}: {description}{value_preview}")


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_filter_extraction(questions: Dict[str, str] = None):
    """Test that filter extraction works correctly."""
    print("=" * 70)
    print("TESTING FILTER EXTRACTION")
    print("=" * 70)

    from src.rag_advanced.filter_utils import extract_multi_filters_from_query, build_chromadb_where_clause

    questions = questions or TEST_QUESTIONS
    for qid, question in questions.items():
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

            # Show expanded values for tsp_normalized
            if 'tsp_normalized' in filters:
                from src.rag_advanced.alias_expander import expand_tsp_aliases
                if isinstance(filters['tsp_normalized'], list):
                    expanded = expand_tsp_aliases(filters['tsp_normalized'])
                    print(f"  Expanded tsp_normalized: {expanded}")


def test_retrieval(question: str, k: int = 15, verbose: bool = False):
    """Test document retrieval for a question."""
    print("\n" + "=" * 70)
    print(f"TESTING RETRIEVAL: {question}")
    print("=" * 70)

    from src.vector_store import get_smart_retriever
    from src.rag_advanced.filter_utils import (
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

        # TSP distribution
        tsps = Counter(d.metadata.get('tsp_normalized', 'unknown') for d in docs)
        print(f"\nTSP distribution:")
        for tsp, count in tsps.most_common():
            print(f"  {tsp}: {count} chunks")

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


def test_all_questions(mode: str = "flash", verbose: bool = False, questions: Dict[str, str] = None):
    """Test all questions with specified mode."""
    questions = questions or TEST_QUESTIONS

    print("\n" + "=" * 70)
    print(f"TESTING ALL QUESTIONS - {mode.upper()} MODE")
    print("=" * 70)

    results = {}
    for qid, question in questions.items():
        # Get metadata if available
        meta = TEST_QUESTIONS_METADATA.get(qid, {})
        role = meta.get("role", "Unknown")
        qtype = meta.get("type", "unknown")

        print(f"\n{'='*70}")
        print(f"{qid} [{role}] ({qtype}): {question}")
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
        meta = TEST_QUESTIONS_METADATA.get(qid, {})
        role = meta.get("role", "")
        print(f"  {qid} [{role:11}]: {status} {result['status']}")

    return results


def test_all_modes(verbose: bool = False, questions: Dict[str, str] = None):
    """Test all questions with BOTH flash and thinking modes."""
    questions = questions or TEST_QUESTIONS

    print("\n" + "=" * 70)
    print("TESTING ALL QUESTIONS - BOTH MODES")
    print("=" * 70)

    all_results = {}

    for mode in ["flash", "thinking"]:
        print(f"\n\n{'#'*70}")
        print(f"# {mode.upper()} MODE")
        print(f"{'#'*70}")
        all_results[mode] = test_all_questions(mode=mode, verbose=verbose, questions=questions)

    # Final comparison
    print("\n\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Question':<12} {'Role':<12} {'Flash':<10} {'Thinking':<10}")
    print("-" * 50)
    for qid in questions.keys():
        meta = TEST_QUESTIONS_METADATA.get(qid, {})
        role = meta.get("role", "")[:10]
        flash_status = "✅" if all_results["flash"].get(qid, {}).get("status") == "OK" else "❌"
        think_status = "✅" if all_results["thinking"].get(qid, {}).get("status") == "OK" else "❌"
        print(f"{qid:<12} {role:<12} {flash_status:<10} {think_status:<10}")

    return all_results


def list_questions():
    """List all available questions with metadata."""
    print("\n" + "=" * 70)
    print("AVAILABLE TEST QUESTIONS")
    print("=" * 70)

    current_role = None
    for qid, meta in TEST_QUESTIONS_METADATA.items():
        role = meta["role"]
        if role != current_role:
            print(f"\n--- {role.upper()} ---")
            current_role = role

        qtype = meta["type"]
        notes = meta.get("notes", "")
        print(f"  {qid}: [{qtype:12}] {meta['question'][:50]}...")
        if notes:
            print(f"       Note: {notes}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test ERCOT RAG - Core tests + C-Suite demo questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_questions.py --all                    # Run all questions
  python demo_questions.py --all --mode both        # Both modes, all questions
  python demo_questions.py --role ceo               # CEO questions only
  python demo_questions.py --role cfo --mode both   # CFO questions, both modes
  python demo_questions.py -t Q5 -v                 # Single question, verbose
  python demo_questions.py --analytics-only         # Test analytics questions
  python demo_questions.py --list                   # List all questions
        """
    )

    # Mode selection
    parser.add_argument("--mode", choices=["flash", "thinking", "both"], default="flash",
                        help="RAG mode to use (default: flash)")

    # Question selection
    parser.add_argument("--question", "-q", type=str,
                        help="Custom question to test")
    parser.add_argument("--test", "-t", type=str,
                        help="Run specific test question (Q1-Q18)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Run ALL test questions")
    parser.add_argument("--role", choices=list(ROLES.keys()),
                        help="Run questions for specific role (ceo, cfo, legal, development, tech, core)")

    # Special modes
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output (stream full response)")
    parser.add_argument("--filters-only", action="store_true",
                        help="Only test filter extraction (no LLM calls)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Only test retrieval (no LLM generation)")
    parser.add_argument("--analytics-only", action="store_true",
                        help="Only test analytics-type questions")
    parser.add_argument("--analytics-check", action="store_true",
                        help="Check analytics data availability")
    parser.add_argument("--list", action="store_true",
                        help="List all available questions")

    args = parser.parse_args()

    # List questions
    if args.list:
        list_questions()
        return

    # Analytics check
    if args.analytics_check:
        test_analytics_availability()
        return

    # Build question set based on filters
    questions = TEST_QUESTIONS.copy()

    if args.role:
        target_role = ROLES[args.role]
        questions = {
            qid: meta["question"]
            for qid, meta in TEST_QUESTIONS_METADATA.items()
            if meta["role"] == target_role
        }
        if not questions:
            print(f"No questions found for role: {args.role}")
            return

    if args.analytics_only:
        questions = {
            qid: meta["question"]
            for qid, meta in TEST_QUESTIONS_METADATA.items()
            if meta["type"] == "analytics"
        }
        if not questions:
            print("No analytics questions found")
            return

    # Filter extraction test
    if args.filters_only:
        test_filter_extraction(questions)
        return

    # Run all questions
    if args.all or args.role or args.analytics_only:
        if args.mode == "both":
            test_all_modes(verbose=args.verbose, questions=questions)
        else:
            test_all_questions(mode=args.mode, verbose=args.verbose, questions=questions)
        return

    # Determine single question to test
    if args.question:
        question = args.question
    elif args.test:
        if args.test not in TEST_QUESTIONS:
            print(f"❌ Question {args.test} not found. Valid: {', '.join(TEST_QUESTIONS.keys())}")
            return
        question = TEST_QUESTIONS[args.test]
    else:
        # Default: show help and list questions
        parser.print_help()
        list_questions()
        return

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

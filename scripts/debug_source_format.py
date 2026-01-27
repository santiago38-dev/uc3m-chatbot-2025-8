#!/usr/bin/env python3
"""
Debug script to verify source formatting matches parser expectations.
Tests the format_citations -> parser pipeline directly.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_rag_sources(generated_answer: str):
    """Copy of the parser from test_utils.py"""
    sources_split = generated_answer.split("Sources:\n", 1)
    main_response = sources_split[0].strip()
    sources_content = sources_split[1].strip() if len(sources_split) > 1 else None
    sources = {"keys": [], "coords": []}

    if not sources_content:
        print(f"[DEBUG] No sources content found after split")
        print(f"[DEBUG] Full answer snippet: {generated_answer[-200:]!r}")
        return main_response, sources

    lines = sources_content.strip().split('\n')
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
        # Parse metadata: [1] Project (INR) - Section
        match = re.search(r'(?:\[\d+\]\s*)?(.*?)\s*\((.*?)\)\s*-\s*(.*)', clean_line)
        if not match:
            print(f"[DEBUG] Line didn't match regex: {clean_line!r}")
            continue

        project = match.group(1).strip()
        inr = match.group(2).strip()
        section = match.group(3).strip()
        key = f"{project}::{inr}::{section}"
        sources["keys"].append(key)
        sources["coords"].append({"project": project, "inr": inr, "section": section})
        print(f"[DEBUG] Parsed: {key}")

    return main_response, sources


def main():
    print("=" * 70)
    print("SOURCE FORMAT DEBUG")
    print("=" * 70)

    # Test 1: Direct format_citations output test
    print("\n[TEST 1] Testing format_citations output format")
    print("-" * 50)

    from src.rag_advanced.utils import format_citations

    # Simulate source data
    test_sources = [
        {'ref': 1, 'project_name': 'Parliament Solar', 'inr': '23INR0044', 'section': 'article_1'},
        {'ref': 2, 'project_name': 'Parliament Solar', 'inr': '23INR0044', 'section': 'article_10'},
        {'ref': 3, 'project_name': 'Parliament Solar', 'inr': '23INR0044', 'section': 'exhibit_c'},
    ]

    citations = format_citations(test_sources)
    print(f"\nformat_citations output (repr):")
    print(repr(citations))
    print(f"\nformat_citations output (rendered):")
    print(citations)

    # Test 2: Parse the formatted output
    print("\n[TEST 2] Testing parser on formatted output")
    print("-" * 50)

    # Create a mock full answer
    mock_answer = "This is the answer.\n\n" + citations

    main_response, parsed = parse_rag_sources(mock_answer)
    print(f"\nParsed keys: {parsed['keys']}")

    expected_keys = [
        "Parliament Solar::23INR0044::article_1",
        "Parliament Solar::23INR0044::article_10",
        "Parliament Solar::23INR0044::exhibit_c"
    ]

    matches = set(parsed['keys']) & set(expected_keys)
    print(f"\nExpected keys: {expected_keys}")
    print(f"Matched keys: {list(matches)}")
    print(f"Match rate: {len(matches)}/{len(expected_keys)} = {len(matches)/len(expected_keys)*100:.1f}%")

    # Test 3: Check format edge cases
    print("\n[TEST 3] Testing edge case formats")
    print("-" * 50)

    edge_cases = [
        "Sources:\n  [1] Project Name (INR123) - section_name",
        "Sources:\n[1] Project Name (INR123) - section_name",  # No leading space
        "Sources:\n  [1] Project Name (INR-123) - section_name",  # Hyphen in INR
        "Sources:\n  Project Name (INR123) - section_name",  # No [N] prefix
    ]

    for case in edge_cases:
        _, parsed = parse_rag_sources(case)
        status = "✅" if parsed['keys'] else "❌"
        print(f"{status} Format: {case.split(chr(10))[1][:40]!r}...")
        if parsed['keys']:
            print(f"    Parsed: {parsed['keys']}")

    # Test 4: Real format from utils.py
    print("\n[TEST 4] Testing actual format_sources chain")
    print("-" * 50)

    from src.rag_advanced.utils import format_sources
    from langchain_core.documents import Document

    # Create mock documents with metadata
    mock_docs = [
        Document(
            page_content="Test content 1",
            metadata={
                'project_name': 'Parliament Solar',
                'inr': '23INR0044',
                'section': 'article_1',
                'section_type': 'security_amounts'
            }
        ),
        Document(
            page_content="Test content 2",
            metadata={
                'project_name': 'Parliament Solar',
                'inr': '23INR0044',
                'section': 'exhibit_c',
                'section_type': 'milestones_timeline'
            }
        ),
    ]

    retrieval = format_sources(mock_docs)
    citations = format_citations(retrieval['sources'])

    print(f"format_sources metadata:")
    for s in retrieval['sources']:
        print(f"  - {s['project_name']} ({s['inr']}) - {s['section']}")

    mock_answer = "Answer text.\n\n" + citations
    _, parsed = parse_rag_sources(mock_answer)

    print(f"\nParsed back:")
    for key in parsed['keys']:
        print(f"  - {key}")

    # Verify round-trip
    expected = {"Parliament Solar::23INR0044::article_1", "Parliament Solar::23INR0044::exhibit_c"}
    actual = set(parsed['keys'])
    if expected == actual:
        print("\n✅ Round-trip format test PASSED")
    else:
        print(f"\n❌ Round-trip format test FAILED")
        print(f"   Expected: {expected}")
        print(f"   Got: {actual}")

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()

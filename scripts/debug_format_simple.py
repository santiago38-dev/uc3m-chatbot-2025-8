#!/usr/bin/env python3
"""
Simple format test - no external dependencies.
"""

import re


def format_citations(sources: list) -> str:
    """Copy of format_citations from utils.py"""
    if not sources:
        return ""
    lines = ["\n\nSources:"]
    for s in sources:
        lines.append(f"  [{s['ref']}] {s['project_name']} ({s['inr']}) - {s['section']}")
    return "\n".join(lines)


def parse_rag_sources(generated_answer: str):
    """Copy of parser from test_utils.py"""
    sources_split = generated_answer.split("Sources:\n", 1)
    main_response = sources_split[0].strip()
    sources_content = sources_split[1].strip() if len(sources_split) > 1 else None
    sources = {"keys": [], "coords": []}

    if not sources_content:
        print(f"[DEBUG] No sources content found")
        print(f"[DEBUG] Split result: {sources_split}")
        return main_response, sources

    lines = sources_content.strip().split('\n')
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
        match = re.search(r'(?:\[\d+\]\s*)?(.*?)\s*\((.*?)\)\s*-\s*(.*)', clean_line)
        if not match:
            print(f"[DEBUG] No regex match for: {clean_line!r}")
            continue

        project = match.group(1).strip()
        inr = match.group(2).strip()
        section = match.group(3).strip()
        key = f"{project}::{inr}::{section}"
        sources["keys"].append(key)
        sources["coords"].append({"project": project, "inr": inr, "section": section})

    return main_response, sources


def main():
    print("=" * 60)
    print("FORMAT VERIFICATION TEST")
    print("=" * 60)

    # Test sources
    test_sources = [
        {'ref': 1, 'project_name': 'Parliament Solar', 'inr': '23INR0044', 'section': 'article_1'},
        {'ref': 2, 'project_name': 'Parliament Solar', 'inr': '23INR0044', 'section': 'article_10'},
        {'ref': 3, 'project_name': 'Peyton Creek Wind II', 'inr': '20INR0155', 'section': 'schedule_of'},
    ]

    # Format
    citations = format_citations(test_sources)
    print(f"\n[1] format_citations output (repr):")
    print(f"    {citations!r}")

    print(f"\n[2] format_citations output (rendered):")
    print(citations)

    # Build full answer
    mock_answer = "This is the answer text.\n" + citations
    print(f"\n[3] Full answer (repr):")
    print(f"    {mock_answer!r}")

    # Parse
    print(f"\n[4] Parsing with split on 'Sources:\\n':")
    main_response, parsed = parse_rag_sources(mock_answer)
    print(f"    Main response: {main_response[:50]!r}...")
    print(f"    Parsed keys: {parsed['keys']}")

    # Verify
    expected_keys = [
        "Parliament Solar::23INR0044::article_1",
        "Parliament Solar::23INR0044::article_10",
        "Peyton Creek Wind II::20INR0155::schedule_of"
    ]

    print(f"\n[5] Verification:")
    print(f"    Expected: {expected_keys}")
    print(f"    Got:      {parsed['keys']}")

    if parsed['keys'] == expected_keys:
        print(f"\n✅ FORMAT TEST PASSED - Round-trip works correctly")
    else:
        print(f"\n❌ FORMAT TEST FAILED")
        print(f"    Missing: {set(expected_keys) - set(parsed['keys'])}")
        print(f"    Extra:   {set(parsed['keys']) - set(expected_keys)}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()

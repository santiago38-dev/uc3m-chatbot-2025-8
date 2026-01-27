#!/usr/bin/env python3
"""
Debug script to trace exactly what's being retrieved vs expected.
Shows the full pipeline: question → retrieved docs → parsed keys → comparison.
"""

import os
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.vector_store import get_retriever
from src.rag_advanced.chain import get_rag_chain
from src.rag_advanced.utils import RAGMode
from rag_tests.sample_dataset_complete import SAMPLE_DATASET


def parse_rag_sources(generated_answer: str):
    """Parse sources from RAG output."""
    sources_split = generated_answer.split("Sources:\n", 1)
    main_response = sources_split[0].strip()
    sources_content = sources_split[1].strip() if len(sources_split) > 1 else None
    sources = {"keys": [], "coords": []}

    if not sources_content:
        return main_response, sources

    lines = sources_content.strip().split('\n')
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
        # Parse metadata: [1] Project (INR) - Section
        match = re.search(r'(?:\[\d+\]\s*)?(.*?)\s*\((.*?)\)\s*-\s*(.*)', clean_line)
        if not match:
            print(f"    [DEBUG] Line didn't match regex: {clean_line!r}")
            continue

        project = match.group(1).strip()
        inr = match.group(2).strip()
        section = match.group(3).strip()
        key = f"{project}::{inr}::{section}"
        sources["keys"].append(key)
        sources["coords"].append({"project": project, "inr": inr, "section": section})

    return main_response, sources


def main():
    print("=" * 80)
    print("RETRIEVAL DEBUG - Tracing what's retrieved vs expected")
    print("=" * 80)

    # Initialize RAG chain
    k_docs = 10
    retriever = get_retriever(k_docs=k_docs)
    rag_chain = get_rag_chain(
        retriever=retriever,
        mode=RAGMode.FLASH,
        k_docs=k_docs,
        with_history=True,
        with_summary=False
    )

    # Test just a few in-scope questions
    test_cases = [case for case in SAMPLE_DATASET if case.get('is_in_scope')][:3]

    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        expected_keys = set(case.get("relevant_doc_keys", []))

        print(f"\n{'=' * 80}")
        print(f"TEST {i}: {question[:70]}...")
        print(f"{'=' * 80}")

        print(f"\n[EXPECTED KEYS] ({len(expected_keys)}):")
        for key in expected_keys:
            print(f"    {key}")

        # Get RAG response
        config = {"configurable": {"session_id": f"debug_{i}"}}
        generated_answer = ""
        for chunk in rag_chain.stream({"question": question}, config=config):
            generated_answer += str(chunk)

        # Show raw sources section
        if "Sources:" in generated_answer:
            sources_section = generated_answer.split("Sources:", 1)[1]
            print(f"\n[RAW SOURCES SECTION]:")
            for line in sources_section.strip().split('\n')[:15]:
                print(f"    {line!r}")

        # Parse sources
        main_response, docs = parse_rag_sources(generated_answer)

        print(f"\n[RETRIEVED KEYS] ({len(docs['keys'])}):")
        for key in docs["keys"]:
            status = "✅" if key in expected_keys else "❌"
            print(f"    {status} {key}")

        # Calculate recall
        retrieved_set = set(docs["keys"])
        matched = expected_keys & retrieved_set
        recall = len(matched) / len(expected_keys) * 100 if expected_keys else 0

        print(f"\n[RESULT]:")
        print(f"    Expected: {len(expected_keys)} keys")
        print(f"    Retrieved: {len(retrieved_set)} unique keys")
        print(f"    Matched: {len(matched)} keys")
        print(f"    Recall: {recall:.1f}%")

        if expected_keys - retrieved_set:
            print(f"\n[MISSING EXPECTED KEYS]:")
            for key in expected_keys - retrieved_set:
                print(f"    ❌ {key}")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

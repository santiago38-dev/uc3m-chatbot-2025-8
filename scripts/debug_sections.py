#!/usr/bin/env python3
"""
Debug script to compare expected document keys vs actual ChromaDB contents.
"""

import os
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.config import Settings

CHROMADB_PATH = os.environ.get(
    'CHROMADB_PATH',
    'C:/Users/szcas/OneDrive/AI Masters/1. ERCOT_NLP/Github/ercot-lgia-rag-system/output/chromadb'
)

def main():
    print("=" * 70)
    print("DEBUG: Expected Keys vs ChromaDB Contents")
    print("=" * 70)

    # Connect to ChromaDB
    print(f"\nConnecting to: {CHROMADB_PATH}")
    client = chromadb.PersistentClient(path=CHROMADB_PATH, settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection("sgia_chunks")
    print(f"Total chunks: {collection.count()}")

    # Projects to check (from sample_dataset)
    projects_to_check = [
        "Parliament Solar",
        "Peyton Creek Wind II",
        "FRIENDSWOOD ENERGY GENCO",
        "Lavaca Bay Solar",
        "Pine Forest BESS",
        "Tanglewood Solar",
        "Myrtle Solar"
    ]

    print("\n" + "=" * 70)
    print("ACTUAL SECTIONS IN CHROMADB PER PROJECT")
    print("=" * 70)

    for project in projects_to_check:
        results = collection.get(
            where={"project_name": {"$eq": project}},
            include=["metadatas"],
            limit=50
        )

        if not results['ids']:
            print(f"\n❌ {project}: NOT FOUND IN CHROMADB!")
            continue

        print(f"\n✅ {project}: {len(results['ids'])} chunks")

        # Count sections
        section_counts = Counter()
        section_type_counts = Counter()

        for meta in results['metadatas']:
            section = meta.get('section', 'N/A')
            section_type = meta.get('section_type', 'N/A')
            section_counts[section] += 1
            section_type_counts[section_type] += 1

        print(f"   'section' field values:")
        for sec, count in section_counts.most_common(10):
            print(f"      {sec}: {count}")

        print(f"   'section_type' field values:")
        for sec, count in section_type_counts.most_common(5):
            print(f"      {sec}: {count}")

    # Now show expected keys from sample_dataset
    print("\n" + "=" * 70)
    print("EXPECTED KEYS IN SAMPLE_DATASET")
    print("=" * 70)

    from rag_tests.sample_dataset_complete import SAMPLE_DATASET

    for case in SAMPLE_DATASET:
        if not case.get('is_in_scope'):
            continue

        keys = case.get('relevant_doc_keys', [])
        if keys:
            print(f"\nQ: {case['question'][:60]}...")
            for key in keys:
                parts = key.split('::')
                if len(parts) == 3:
                    project, inr, section = parts
                    # Check if this section exists
                    results = collection.get(
                        where={
                            "$and": [
                                {"project_name": {"$eq": project}},
                                {"section": {"$eq": section}}
                            ]
                        },
                        limit=1
                    )
                    exists = "✅" if results['ids'] else "❌"
                    print(f"   {exists} {key}")


if __name__ == '__main__':
    main()

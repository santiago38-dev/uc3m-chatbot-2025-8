#!/usr/bin/env python3
"""
Corpus Verification Script - Inspect ChromaDB metadata values.

Run this script to see what values are actually stored in your ChromaDB,
which helps populate the CHROMADB_PARENT_ALIASES and CHROMADB_TSP_ALIASES
mappings in alias_expander.py.

Usage:
    python scripts/verify_corpus_developers.py

Output:
    - List of all unique parent_company values
    - List of all unique tsp_normalized values
    - Specific variants for RWE, SAMSUNG, etc.
"""

import os
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()


def get_chromadb_path() -> str:
    """Get ChromaDB path from environment or use default."""
    return os.getenv("CHROMADB_PATH", "./output/chromadb")


def verify_corpus():
    """Inspect ChromaDB corpus and print metadata analysis."""

    try:
        import chromadb
    except ImportError:
        print("Error: chromadb not installed. Run: pip install chromadb")
        return

    chromadb_path = get_chromadb_path()
    collection_name = os.getenv("COLLECTION_NAME", "sgia_chunks")

    print("=" * 70)
    print("CHROMADB CORPUS VERIFICATION")
    print("=" * 70)
    print(f"Path: {chromadb_path}")
    print(f"Collection: {collection_name}")
    print()

    # Check if path exists
    if not Path(chromadb_path).exists():
        print(f"ERROR: ChromaDB path does not exist: {chromadb_path}")
        print("\nPlease ensure:")
        print("  1. ChromaDB has been created by the chunking pipeline")
        print("  2. CHROMADB_PATH in .env points to the correct location")
        return

    try:
        client = chromadb.PersistentClient(path=chromadb_path)
        collection = client.get_collection(collection_name)
    except Exception as e:
        print(f"ERROR: Failed to load ChromaDB: {e}")
        return

    # Get all metadata
    print("Loading all metadata from corpus...")
    try:
        results = collection.get(include=['metadatas'])
    except Exception as e:
        print(f"ERROR: Failed to query collection: {e}")
        return

    if not results['metadatas']:
        print("WARNING: No documents found in collection!")
        return

    total_docs = len(results['metadatas'])
    print(f"Found {total_docs} documents\n")

    # Analyze metadata
    parent_companies = Counter()
    tsps = Counter()
    zones = Counter()
    fuel_types = Counter()
    projects = Counter()

    for meta in results['metadatas']:
        pc = meta.get('parent_company', '')
        tsp = meta.get('tsp_normalized', '')
        zone = meta.get('zone', '')
        fuel = meta.get('fuel_type', '')
        project = meta.get('project_name', '')

        if pc:
            parent_companies[pc] += 1
        if tsp:
            tsps[tsp] += 1
        if zone:
            zones[zone] += 1
        if fuel:
            fuel_types[fuel] += 1
        if project:
            projects[project] += 1

    # Print results
    print("=" * 70)
    print("PARENT COMPANIES IN CORPUS (top 30)")
    print("=" * 70)
    for company, count in parent_companies.most_common(30):
        print(f"  '{company}': {count} chunks")

    print("\n" + "=" * 70)
    print("TSPs IN CORPUS")
    print("=" * 70)
    for tsp, count in tsps.most_common():
        print(f"  '{tsp}': {count} chunks")

    print("\n" + "=" * 70)
    print("ZONES IN CORPUS")
    print("=" * 70)
    for zone, count in zones.most_common():
        print(f"  '{zone}': {count} chunks")

    print("\n" + "=" * 70)
    print("FUEL TYPES IN CORPUS")
    print("=" * 70)
    for fuel, count in fuel_types.most_common():
        print(f"  '{fuel}': {count} chunks")

    print("\n" + "=" * 70)
    print("UNIQUE PROJECTS")
    print("=" * 70)
    print(f"  Total unique projects: {len(projects)}")

    # === SPECIFIC DEVELOPER ANALYSIS ===
    print("\n" + "=" * 70)
    print("RWE VARIANTS IN CORPUS")
    print("=" * 70)
    rwe_variants = [k for k in parent_companies.keys() if 'RWE' in k.upper()]
    if rwe_variants:
        for v in rwe_variants:
            print(f"  '{v}': {parent_companies[v]} chunks")
    else:
        print("  ⚠️  NO RWE PROJECTS FOUND IN CORPUS!")

    print("\n" + "=" * 70)
    print("SAMSUNG VARIANTS IN CORPUS")
    print("=" * 70)
    samsung_variants = [k for k in parent_companies.keys() if 'SAMSUNG' in k.upper()]
    if samsung_variants:
        for v in samsung_variants:
            print(f"  '{v}': {parent_companies[v]} chunks")
    else:
        print("  ⚠️  NO SAMSUNG PROJECTS FOUND IN CORPUS!")

    print("\n" + "=" * 70)
    print("NEXTERA VARIANTS IN CORPUS")
    print("=" * 70)
    nextera_variants = [k for k in parent_companies.keys()
                        if 'NEXTERA' in k.upper() or 'FPL' in k.upper()]
    if nextera_variants:
        for v in nextera_variants:
            print(f"  '{v}': {parent_companies[v]} chunks")
    else:
        print("  ⚠️  NO NEXTERA PROJECTS FOUND IN CORPUS!")

    # === ONCOR / CENTERPOINT ANALYSIS ===
    print("\n" + "=" * 70)
    print("ONCOR VARIANTS IN CORPUS")
    print("=" * 70)
    oncor_variants = [k for k in tsps.keys() if 'ONCOR' in k.upper()]
    if oncor_variants:
        for v in oncor_variants:
            print(f"  '{v}': {tsps[v]} chunks")
    else:
        print("  ⚠️  NO ONCOR TSP FOUND IN CORPUS!")

    print("\n" + "=" * 70)
    print("CENTERPOINT VARIANTS IN CORPUS")
    print("=" * 70)
    centerpoint_variants = [k for k in tsps.keys()
                            if 'CENTERPOINT' in k.upper() or 'CNP' in k.upper()]
    if centerpoint_variants:
        for v in centerpoint_variants:
            print(f"  '{v}': {tsps[v]} chunks")
    else:
        print("  ⚠️  NO CENTERPOINT TSP FOUND IN CORPUS!")

    # === ALIAS RECOMMENDATIONS ===
    print("\n" + "=" * 70)
    print("RECOMMENDED ALIAS UPDATES")
    print("=" * 70)
    print("\nCopy these values to src/rag_advanced/alias_expander.py:")
    print("\nCHROMADB_PARENT_ALIASES additions:")

    for company in ['RWE', 'SAMSUNG', 'NEXTERA']:
        variants = [k for k in parent_companies.keys()
                    if company in k.upper() or
                    (company == 'NEXTERA' and 'FPL' in k.upper())]
        if variants:
            print(f"\n    '{company}': [")
            for v in variants:
                print(f"        '{v}',")
            print("    ],")

    print("\nCHROMADB_TSP_ALIASES additions:")
    for tsp in ['ONCOR', 'CENTERPOINT', 'AEP']:
        variants = [k for k in tsps.keys()
                    if tsp in k.upper() or
                    (tsp == 'CENTERPOINT' and 'CNP' in k.upper())]
        if variants:
            print(f"\n    '{tsp}': [")
            for v in variants:
                print(f"        '{v}',")
            print("    ],")


def main():
    """Main entry point."""
    verify_corpus()


if __name__ == "__main__":
    main()

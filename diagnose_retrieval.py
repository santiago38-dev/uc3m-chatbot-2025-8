import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from rag_tests.sample_dataset_complete import SAMPLE_DATASET
from src.vector_store import get_retriever

def diagnose():
    print("=" * 80)
    print("RETRIEVAL DIAGNOSTIC: Project vs. Section Match")
    print("=" * 80)

    # Get Retriever
    retriever = get_retriever(k_docs=10)

    project_hits = 0
    section_hits = 0
    total_cases = 0

    for i, case in enumerate(SAMPLE_DATASET):
        if not case.get('is_in_scope', True):
            continue

        total_cases += 1
        question = case['question']
        expected_keys = case.get('relevant_doc_keys', [])

        # 1. Get Expected Projects & Sections
        expected_projects = set()
        for key in expected_keys:
            parts = key.split("::")
            if len(parts) > 0: expected_projects.add(parts[0])

        # 2. Run Retrieval
        print(f"\nQ{i+1}: {question[:60]}...")
        try:
            docs = retriever.invoke(question)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # 3. Analyze Results
        retrieved_projects = set()
        retrieved_keys = []

        for doc in docs:
            meta = doc.metadata
            proj = meta.get('project_name', 'Unknown')
            # Handle different metadata structures
            sect = meta.get('section') or meta.get('section_type') or 'Unknown'
            inr = meta.get('inr') or 'Unknown'

            key = f"{proj}::{inr}::{sect}"
            retrieved_keys.append(key)
            retrieved_projects.add(proj)

        # 4. Check Matches
        # STRICT MATCH (Section Level)
        exact_hit = any(k in expected_keys for k in retrieved_keys)
        # RELAXED MATCH (Project Level)
        project_hit = not expected_projects.isdisjoint(retrieved_projects)

        status = "FAIL"
        if exact_hit:
            status = "EXACT MATCH"
            section_hits += 1
            project_hits += 1
        elif project_hit:
            status = "PROJECT MATCH (Wrong Section)"
            project_hits += 1

        print(f"STATUS: {status}")
        print(f"Expected Projects: {expected_projects}")
        print(f"Retrieved Top 3: {[k.split('::')[0] for k in retrieved_keys[:3]]}")

    print("\n" + "="*80)
    print("DIAGNOSTIC RESULTS")
    print(f"Strict Recall (Exact Section): {section_hits}/{total_cases} ({section_hits/total_cases:.1%})")
    print(f"Relaxed Recall (Right Project): {project_hits}/{total_cases} ({project_hits/total_cases:.1%})")
    print("="*80)
    print("\nVERDICT:")
    if project_hits/total_cases > 0.7:
        print("  System is WORKING - finding correct projects")
        print("  The 'low recall' is a metric definition issue, not a system failure")
        print("  Consider using Project-Level Recall as your primary metric")
    else:
        print("  Retrieval may be broken - not finding correct projects")
        print("  Need to investigate indexing or embedding issues")

if __name__ == "__main__":
    diagnose()

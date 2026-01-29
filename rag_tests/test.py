"""
ERCOT SGIA RAG System - Complete Test Dataset
Generated for Santiago's UC3M Applied AI Master's Thesis

This dataset provides comprehensive coverage across:
- Languages: English and Spanish
- Fuel types: SOL (Solar), WIN (Wind), OTH (Battery/BESS), GAS
- Zones: COAST, NORTH, SOUTH, WEST, PANHANDLE
- Question types: Security, TSP, Contacts, Capacity, Milestones, Comparative, Zone-based, Fuel-type
- Scope: In-scope (ERCOT) and Out-of-scope questions

Dataset Statistics:
- Total test cases: 24
- Out-of-scope English: 3
- Out-of-scope Spanish: 3
- In-scope English: 14
- In-scope Spanish: 4

"""
import sys

from .sample_dataset_complete import SAMPLE_DATASET

from rag_tests.test_utils import RAGQualityEvaluator, run_evaluation, validate_dataset, print_coverage_stats
from src.rag_advanced.utils import RAGMode

# Import for document key generation
# Use this helper: RAGQualityEvaluator.doc_to_key(project_name, inr, section)



# =============================================================================
# COVERAGE SUMMARY
# =============================================================================
"""
COVERAGE SUMMARY

LANGUAGE COVERAGE:
├── Out-of-scope English: 3 questions
├── Out-of-scope Spanish: 3 questions
├── In-scope English: 14 questions
└── In-scope Spanish: 4 questions
TOTAL: 24 test cases

FUEL TYPE COVERAGE:
├── SOL (Solar): 8 questions
│   ├── Parliament Solar (COAST/CENTERPOINT)
│   ├── Dorado Solar (WEST/LONE STAR)
│   ├── Tanglewood Solar (COAST/CENTERPOINT)
│   ├── Lavaca Bay Solar (COAST/CENTERPOINT)
│   ├── Myrtle Solar (COAST/CENTERPOINT)
│   ├── Stoneridge Solar (SOUTH/ONCOR)
│   └── Pine Forest Solar mentioned
├── WIN (Wind): 3 questions
│   ├── Peyton Creek Wind II (COAST/CENTERPOINT)
│   ├── Fortuna Wind (NORTH/ONCOR)
│   └── Lane City Wind referenced
├── OTH (Battery/BESS): 6 questions
│   ├── Houston IV BESS (COAST/CENTERPOINT)
│   ├── Pine Forest BESS (NORTH/ONCOR)
│   ├── Acker BESS (PANHANDLE/ONCOR)
│   └── Champaign BESS (WEST/ONCOR)
└── GAS (Natural Gas): 3 questions
    ├── FRIENDSWOOD ENERGY GENCO (COAST/CENTERPOINT)
    ├── Remy Jade III Power Station (COAST/CENTERPOINT)
    └── Cedar Bayou 5, NRG Greens Bayou 6 referenced

ZONE COVERAGE:
├── COAST: 12 questions (dominant - matches corpus distribution)
├── NORTH: 5 questions
├── SOUTH: 2 questions
├── WEST: 3 questions
└── PANHANDLE: 1 question

TSP COVERAGE:
├── CENTERPOINT: 14 questions
├── ONCOR: 7 questions
├── LONE STAR: 2 questions
├── AEP: Referenced
└── LCRA: Referenced

SECTION COVERAGE:
├── exhibit_e (Security): 14 references
├── exhibit_a (Facility): 8 references
├── schedule_of (Contacts): 6 references
└── exhibit_b (Milestones): Covered via exhibit_a

QUESTION TYPE COVERAGE:
├── Security Amount: 6 questions (EN: 4, ES: 2)
├── TSP Identification: 4 questions (EN: 2, ES: 2)
├── Contact/Email: 2 questions (EN)
├── Capacity/Facility: 4 questions (EN: 2, ES: 2)
├── Comparative: 2 questions (EN)
├── Zone/Fuel Filtering: 2 questions (EN)
└── Out-of-scope: 6 questions (EN: 3, ES: 3)
"""


if __name__ == "__main__":
    mode_str = sys.argv[1] if len(sys.argv) > 1 else "flash"

    print("Validating SAMPLE_DATASET...")
    validate_dataset(SAMPLE_DATASET)
    print_coverage_stats(SAMPLE_DATASET)

    if mode_str.lower() == "both":
        print("\n" + "="*60)
        print("RUNNING FLASH MODE")
        print("="*60)
        results_flash = run_evaluation(SAMPLE_DATASET, mode=RAGMode.FLASH)

        print("\n" + "="*60)
        print("RUNNING THINKING MODE")
        print("="*60)
        results_thinking = run_evaluation(SAMPLE_DATASET, mode=RAGMode.THINKING)

        print("\nBoth evaluations complete!")
    else:
        mode = RAGMode.FLASH if mode_str.lower() == "flash" else RAGMode.THINKING
        results = run_evaluation(SAMPLE_DATASET, mode=mode)
        print("\nEvaluation complete!")

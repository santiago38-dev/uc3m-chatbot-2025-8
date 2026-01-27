#!/usr/bin/env python3
"""
Truncation Risk Analysis for ERCOT RAG System

Tests whether all-MiniLM-L6-v2's 256 token limit is causing data loss.
This settles the Claude vs Gemini debate on chunk size empirically.

Decision Matrix:
  - <10% critical loss: Keep current index (Claude wins)
  - 10-20% critical loss: Consider re-index at 650 chars
  - >20% critical loss: Re-index at 600 chars (Gemini wins)

Usage:
    python scripts/test_truncation.py
"""

import os
import sys
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers")
    sys.exit(1)

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ERROR: chromadb not installed. Run: pip install chromadb")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================
CHROMADB_PATH = os.environ.get(
    'CHROMADB_PATH',
    str(Path(__file__).parent.parent / 'output' / 'chromadb')
)
COLLECTION_NAME = os.environ.get('COLLECTION_NAME', 'sgia_chunks')
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MAX_TOKENS = 256  # Hard limit of all-MiniLM-L6-v2


def main():
    print("=" * 70)
    print("TRUNCATION RISK ANALYSIS")
    print("Settling the Claude vs Gemini debate with empirical data")
    print("=" * 70)

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"      Loaded: {MODEL_NAME}")
        print(f"      Max sequence length: {MAX_TOKENS} tokens")
    except Exception as e:
        print(f"ERROR: Could not load tokenizer: {e}")
        sys.exit(1)

    # Load ChromaDB
    print(f"\n[2/4] Connecting to ChromaDB...")
    print(f"      Path: {CHROMADB_PATH}")

    if not Path(CHROMADB_PATH).exists():
        print(f"ERROR: ChromaDB path does not exist: {CHROMADB_PATH}")
        print("       Set CHROMADB_PATH environment variable to your ChromaDB location")
        sys.exit(1)

    try:
        client = chromadb.PersistentClient(
            path=CHROMADB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(COLLECTION_NAME)
        total_count = collection.count()
        print(f"      Collection: {COLLECTION_NAME}")
        print(f"      Total chunks: {total_count}")
    except Exception as e:
        print(f"ERROR: Could not connect to ChromaDB: {e}")
        sys.exit(1)

    # Sample chunks
    sample_size = min(500, total_count)
    print(f"\n[3/4] Analyzing {sample_size} chunks...")

    results = collection.get(
        limit=sample_size,
        include=['documents', 'metadatas']
    )

    # Analysis counters
    stats = {
        'total_analyzed': 0,
        'truncated': 0,
        'safe': 0,
        'lost_dollar_amounts': 0,
        'lost_mw_values': 0,
        'lost_dates': 0,
        'lost_exhibit_refs': 0,
        'lost_article_refs': 0,
        'lost_section_refs': 0,
    }

    critical_losses = []
    token_counts = []
    section_losses = Counter()

    for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
        if doc is None:
            continue

        stats['total_analyzed'] += 1

        # Tokenize
        tokens = tokenizer.encode(doc, add_special_tokens=True)
        token_counts.append(len(tokens))

        if len(tokens) <= MAX_TOKENS:
            stats['safe'] += 1
            continue

        stats['truncated'] += 1

        # Decode what's LOST (tokens beyond limit)
        truncated_tokens = tokens[MAX_TOKENS:]
        lost_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        # Check what important content is lost
        lost_dollar = '$' in lost_text
        lost_mw = 'MW' in lost_text.upper() or 'MEGAWATT' in lost_text.upper()
        lost_date = any(str(year) in lost_text for year in range(2020, 2035))
        lost_exhibit = 'EXHIBIT' in lost_text.upper()
        lost_article = 'ARTICLE' in lost_text.upper()
        lost_section = 'SECTION' in lost_text.upper()

        if lost_dollar:
            stats['lost_dollar_amounts'] += 1
        if lost_mw:
            stats['lost_mw_values'] += 1
        if lost_date:
            stats['lost_dates'] += 1
        if lost_exhibit:
            stats['lost_exhibit_refs'] += 1
        if lost_article:
            stats['lost_article_refs'] += 1
        if lost_section:
            stats['lost_section_refs'] += 1

        # Track by section type
        section_type = meta.get('section_type', 'Unknown') if meta else 'Unknown'
        if lost_dollar or lost_mw:
            section_losses[section_type] += 1

        # Track critical losses for review
        if lost_dollar or lost_mw:
            critical_losses.append({
                'project': meta.get('project_name', 'Unknown') if meta else 'Unknown',
                'section': section_type,
                'inr': meta.get('inr', 'Unknown') if meta else 'Unknown',
                'total_tokens': len(tokens),
                'lost_tokens': len(truncated_tokens),
                'lost_dollar': lost_dollar,
                'lost_mw': lost_mw,
                'lost_preview': lost_text[:150].replace('\n', ' ')
            })

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"      Processed {i + 1}/{sample_size}...")

    # ==========================================================================
    # RESULTS
    # ==========================================================================

    print("\n[4/4] Results")
    print("=" * 70)

    if stats['total_analyzed'] == 0:
        print("ERROR: No chunks were analyzed!")
        sys.exit(1)

    truncation_rate = (stats['truncated'] / stats['total_analyzed']) * 100

    print(f"\nCHUNK STATISTICS:")
    print(f"   Total analyzed: {stats['total_analyzed']}")
    print(f"   Safe (≤{MAX_TOKENS} tokens): {stats['safe']} ({100-truncation_rate:.1f}%)")
    print(f"   Truncated (>{MAX_TOKENS} tokens): {stats['truncated']} ({truncation_rate:.1f}%)")

    print(f"\nCONTENT LOSS IN TRUNCATED CHUNKS:")
    if stats['truncated'] > 0:
        print(f"   Lost $ amounts: {stats['lost_dollar_amounts']} ({stats['lost_dollar_amounts']/stats['truncated']*100:.1f}% of truncated)")
        print(f"   Lost MW values: {stats['lost_mw_values']} ({stats['lost_mw_values']/stats['truncated']*100:.1f}% of truncated)")
        print(f"   Lost dates (2020-2035): {stats['lost_dates']}")
        print(f"   Lost Exhibit refs: {stats['lost_exhibit_refs']}")
        print(f"   Lost Article refs: {stats['lost_article_refs']}")
        print(f"   Lost Section refs: {stats['lost_section_refs']}")
    else:
        print("   No truncation detected!")

    if token_counts:
        print(f"\nTOKEN DISTRIBUTION:")
        print(f"   Min tokens: {min(token_counts)}")
        print(f"   Max tokens: {max(token_counts)}")
        print(f"   Avg tokens: {sum(token_counts)/len(token_counts):.1f}")
        sorted_counts = sorted(token_counts)
        print(f"   Median tokens: {sorted_counts[len(sorted_counts)//2]}")
        print(f"   95th percentile: {sorted_counts[int(len(sorted_counts)*0.95)]}")

    if section_losses:
        print(f"\nCRITICAL LOSSES BY SECTION TYPE:")
        for section, count in section_losses.most_common(10):
            print(f"   {section}: {count}")

    # ==========================================================================
    # VERDICT
    # ==========================================================================

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    critical_loss_count = stats['lost_dollar_amounts'] + stats['lost_mw_values']
    critical_loss_rate = critical_loss_count / stats['total_analyzed'] * 100

    if critical_loss_rate < 5:
        verdict = "LOW RISK"
        symbol = "✅"
        recommendation = """
   Current chunk size is acceptable.
   Hard metadata filters protect numeric queries.
   Recommendation: KEEP CURRENT INDEX.

   >>> CLAUDE WINS - No re-indexing needed <<<"""
    elif critical_loss_rate < 15:
        verdict = "MODERATE RISK"
        symbol = "⚠️"
        recommendation = """
   Some important data is being truncated.
   Consider the cost/benefit of re-indexing.
   Recommendation: RE-INDEX AT 650 CHARS if time permits.

   >>> DRAW - Test specific queries to decide <<<"""
    else:
        verdict = "HIGH RISK"
        symbol = "🚨"
        recommendation = """
   Significant data loss detected.
   Recommendation: RE-INDEX AT 600 CHARS.

   >>> GEMINI WINS - Re-indexing recommended <<<"""

    print(f"\n{symbol} {verdict} ({critical_loss_rate:.1f}% critical loss rate)")
    print(recommendation)

    # Show examples of critical losses
    if critical_losses:
        print("\n" + "=" * 70)
        print(f"SAMPLE CRITICAL LOSSES (showing {min(10, len(critical_losses))} of {len(critical_losses)})")
        print("=" * 70)

        for item in critical_losses[:10]:
            print(f"\n  Project: {item['project']} ({item['inr']})")
            print(f"  Section: {item['section']}")
            print(f"  Tokens: {item['total_tokens']} (lost {item['lost_tokens']})")
            print(f"  Lost $: {'YES' if item['lost_dollar'] else 'no'} | Lost MW: {'YES' if item['lost_mw'] else 'no'}")
            print(f"  Preview: \"{item['lost_preview'][:100]}...\"")

    # Save report
    report_path = Path(__file__).parent.parent / "truncation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("TRUNCATION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total chunks analyzed: {stats['total_analyzed']}\n")
        f.write(f"Truncated: {stats['truncated']} ({truncation_rate:.1f}%)\n")
        f.write(f"Lost $ amounts: {stats['lost_dollar_amounts']}\n")
        f.write(f"Lost MW values: {stats['lost_mw_values']}\n")
        f.write(f"Critical loss rate: {critical_loss_rate:.1f}%\n")
        f.write(f"\nVERDICT: {verdict}\n\n")

        f.write("Critical Losses Detail:\n")
        f.write("-" * 50 + "\n")
        for item in critical_losses:
            f.write(f"\n{item['project']} ({item['section']})\n")
            f.write(f"Tokens: {item['total_tokens']}, Lost: {item['lost_tokens']}\n")
            f.write(f"Lost content: {item['lost_preview']}\n")

    print(f"\n📝 Full report saved to: {report_path}")
    print("=" * 70)

    # Return exit code based on verdict
    if critical_loss_rate >= 15:
        sys.exit(2)  # High risk
    elif critical_loss_rate >= 5:
        sys.exit(1)  # Moderate risk
    else:
        sys.exit(0)  # Low risk


if __name__ == '__main__':
    main()

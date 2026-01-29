"""
Corpus Analytics Generator for ERCOT RAG System

Pre-computes corpus-wide statistics that RAG cannot compute at query time.
These statistics enable answering aggregation questions like:
- "What's the median security cost per kW?"
- "Rank TSPs by average security requirement"
- "Which developers have projects in multiple zones?"

Usage:
    python -m src.analytics.corpus_analytics --output data/corpus_analytics.json
"""

import argparse
import json
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

# Default ChromaDB path (relative to repo root) - matches vector_store.py
DEFAULT_CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./output/chromadb")
DEFAULT_OUTPUT_PATH = "data/corpus_analytics.json"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sgia_chunks")

# =============================================================================
# KEY DEVELOPERS - Externalized configuration for tracked developers
# Add new developers here to include them in specific_developers analytics
# =============================================================================
KEY_DEVELOPERS = ["RWE", "NEXTERA", "SAMSUNG"]

# Minimum pattern length for fuzzy matching (prevents false positives)
MIN_PATTERN_LENGTH = 3

# Fuel type code mapping for display
FUEL_TYPE_DISPLAY = {
    "WIN": "Wind",
    "SOL": "Solar",
    "OTH": "Battery",  # OTH is primarily battery storage
    "GAS": "Gas",
    "UNKNOWN": "Unknown"
}

# Minimum sample size for data quality flagging
MIN_SAMPLE_SIZE = 10


def connect_to_chromadb(chromadb_path: str) -> chromadb.Collection:
    """Connect to ChromaDB and return the collection."""
    client = chromadb.PersistentClient(
        path=chromadb_path,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


def get_all_documents(collection: chromadb.Collection) -> List[Dict[str, Any]]:
    """Retrieve all documents from the collection with metadata."""
    # Get total count first
    count = collection.count()

    # Retrieve all documents (ChromaDB returns in batches)
    results = collection.get(
        include=["metadatas", "documents"],
        limit=count
    )

    documents = []
    for i, doc_id in enumerate(results["ids"]):
        metadata = results["metadatas"][i] if results["metadatas"] else {}
        content = results["documents"][i] if results["documents"] else ""
        documents.append({
            "id": doc_id,
            "metadata": metadata,
            "content": content
        })

    return documents


def deduplicate_by_inr(documents: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Deduplicate documents by INR, keeping the chunk with security_per_kw data.

    Strategy: Group by INR, prefer chunks that have security_per_kw data.
    If multiple chunks have security_per_kw, take the one with highest value
    (as a proxy for "most complete" data).

    Returns:
        Dict mapping INR -> best document metadata
    """
    inr_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for doc in documents:
        inr = doc["metadata"].get("inr")
        if inr:
            inr_groups[inr].append(doc)

    deduplicated = {}
    for inr, chunks in inr_groups.items():
        # Sort chunks: prefer those with security_per_kw, then by highest value
        def chunk_score(chunk: Dict[str, Any]) -> Tuple[int, float]:
            meta = chunk["metadata"]
            has_security = meta.get("security_per_kw") is not None
            security_val = float(meta.get("security_per_kw", 0) or 0)
            return (1 if has_security else 0, security_val)

        best_chunk = max(chunks, key=chunk_score)
        deduplicated[inr] = best_chunk["metadata"]

    return deduplicated


def compute_statistics(values: List[float]) -> Dict[str, Any]:
    """Compute descriptive statistics for a list of values."""
    if not values:
        return {
            "n": 0,
            "median": None,
            "mean": None,
            "min": None,
            "max": None,
            "std_dev": None
        }

    n = len(values)
    result = {
        "n": n,
        "median": round(statistics.median(values), 2),
        "mean": round(statistics.mean(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }

    # Standard deviation requires at least 2 values
    if n >= 2:
        result["std_dev"] = round(statistics.stdev(values), 2)
    else:
        result["std_dev"] = None

    return result


def generate_corpus_analytics(chromadb_path: str = DEFAULT_CHROMADB_PATH) -> Dict[str, Any]:
    """
    Generate comprehensive corpus analytics from ChromaDB.

    Returns:
        Dictionary containing all analytics metrics
    """
    # Connect and retrieve data
    collection = connect_to_chromadb(chromadb_path)
    documents = get_all_documents(collection)

    # Deduplicate by INR
    projects = deduplicate_by_inr(documents)
    total_projects = len(projects)

    # Collect security_per_kw values
    security_values = []
    projects_with_security = []

    for inr, meta in projects.items():
        security_per_kw = meta.get("security_per_kw")
        if security_per_kw is not None:
            try:
                val = float(security_per_kw)
                if val > 0:  # Filter out invalid/zero values
                    security_values.append(val)
                    projects_with_security.append((inr, meta, val))
            except (ValueError, TypeError):
                pass

    # --- CORPUS STATS ---
    corpus_stats = {
        "total_projects": total_projects,
        "total_chunks": len(documents),
        "projects_with_security_data": len(projects_with_security),
        "security_per_kw": compute_statistics(security_values)
    }

    # --- BY FUEL TYPE ---
    fuel_type_data: Dict[str, List[float]] = defaultdict(list)
    fuel_type_counts: Dict[str, int] = defaultdict(int)

    for inr, meta in projects.items():
        fuel_code = meta.get("fuel_type", "UNKNOWN") or "UNKNOWN"
        fuel_type_counts[fuel_code] += 1

        security_per_kw = meta.get("security_per_kw")
        if security_per_kw is not None:
            try:
                val = float(security_per_kw)
                if val > 0:
                    fuel_type_data[fuel_code].append(val)
            except (ValueError, TypeError):
                pass

    by_fuel_type = {}
    for fuel_code, values in fuel_type_data.items():
        display_name = FUEL_TYPE_DISPLAY.get(fuel_code, fuel_code)
        stats = compute_statistics(values)
        by_fuel_type[display_name] = {
            "count": fuel_type_counts.get(fuel_code, 0),
            "projects_with_security_data": stats["n"],
            "median_security_per_kw": stats["median"],
            "mean_security_per_kw": stats["mean"],
            "min_security_per_kw": stats["min"],  # Q10 fix: add min/max ranges
            "max_security_per_kw": stats["max"],  # Q10 fix: add min/max ranges
            "n": stats["n"]
        }

    # Include fuel types with no security data
    for fuel_code, count in fuel_type_counts.items():
        display_name = FUEL_TYPE_DISPLAY.get(fuel_code, fuel_code)
        if display_name not in by_fuel_type:
            by_fuel_type[display_name] = {
                "count": count,
                "projects_with_security_data": 0,
                "median_security_per_kw": None,
                "mean_security_per_kw": None,
                "min_security_per_kw": None,
                "max_security_per_kw": None,
                "n": 0
            }

    # --- BY ZONE ---
    zone_data: Dict[str, List[float]] = defaultdict(list)
    zone_counts: Dict[str, int] = defaultdict(int)

    for inr, meta in projects.items():
        zone = meta.get("zone", "UNKNOWN") or "UNKNOWN"
        zone_counts[zone] += 1

        security_per_kw = meta.get("security_per_kw")
        if security_per_kw is not None:
            try:
                val = float(security_per_kw)
                if val > 0:
                    zone_data[zone].append(val)
            except (ValueError, TypeError):
                pass

    by_zone = {}
    for zone in set(list(zone_data.keys()) + list(zone_counts.keys())):
        values = zone_data.get(zone, [])
        stats = compute_statistics(values)
        by_zone[zone] = {
            "count": zone_counts.get(zone, 0),
            "projects_with_security_data": stats["n"],
            "median_security_per_kw": stats["median"],
            "mean_security_per_kw": stats["mean"],
            "n": stats["n"]
        }

    # --- TSP RANKINGS ---
    tsp_data: Dict[str, List[float]] = defaultdict(list)
    tsp_counts: Dict[str, int] = defaultdict(int)

    for inr, meta in projects.items():
        tsp = meta.get("tsp_normalized", "UNKNOWN") or "UNKNOWN"
        tsp_counts[tsp] += 1

        security_per_kw = meta.get("security_per_kw")
        if security_per_kw is not None:
            try:
                val = float(security_per_kw)
                if val > 0:
                    tsp_data[tsp].append(val)
            except (ValueError, TypeError):
                pass

    tsp_rankings = []
    for tsp, values in tsp_data.items():
        if values:  # Only include TSPs with security data
            stats = compute_statistics(values)
            tsp_rankings.append({
                "tsp": tsp,
                "avg_security_per_kw": stats["mean"],
                "median_security_per_kw": stats["median"],
                "project_count": tsp_counts.get(tsp, 0),
                "n": stats["n"]
            })

    # Sort by average security (descending)
    tsp_rankings.sort(key=lambda x: x["avg_security_per_kw"] or 0, reverse=True)

    # Add rank
    for i, item in enumerate(tsp_rankings, 1):
        item["rank"] = i

    # --- DEVELOPER ANALYSIS ---
    developer_zones: Dict[str, set] = defaultdict(set)
    developer_fuels: Dict[str, set] = defaultdict(set)
    developer_projects: Dict[str, int] = defaultdict(int)
    developer_security: Dict[str, List[float]] = defaultdict(list)

    for inr, meta in projects.items():
        developer = meta.get("parent_company")
        if not developer:
            continue

        developer_projects[developer] += 1

        zone = meta.get("zone")
        if zone:
            developer_zones[developer].add(zone)

        fuel = meta.get("fuel_type")
        if fuel:
            developer_fuels[developer].add(FUEL_TYPE_DISPLAY.get(fuel, fuel))

        security_per_kw = meta.get("security_per_kw")
        if security_per_kw is not None:
            try:
                val = float(security_per_kw)
                if val > 0:
                    developer_security[developer].append(val)
            except (ValueError, TypeError):
                pass

    # Multi-zone developers (presence in 2+ zones)
    multi_zone_developers = {
        dev: sorted(list(zones))
        for dev, zones in developer_zones.items()
        if len(zones) >= 2
    }

    # Diversified portfolios (2+ technologies)
    diversified_portfolios = []
    for dev, fuels in developer_fuels.items():
        if len(fuels) >= 2:
            diversified_portfolios.append({
                "developer": dev,
                "technologies": sorted(list(fuels)),
                "project_count": developer_projects.get(dev, 0),
                "n": developer_projects.get(dev, 0)
            })

    # Sort by project count
    diversified_portfolios.sort(key=lambda x: x["project_count"], reverse=True)

    developer_analysis = {
        "multi_zone_developers": multi_zone_developers,
        "diversified_portfolios": diversified_portfolios
    }

    # --- GEOGRAPHIC CONCENTRATION ---
    county_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "zone": None})

    for inr, meta in projects.items():
        county = meta.get("county")
        if county:
            county_data[county]["count"] += 1
            if not county_data[county]["zone"]:
                county_data[county]["zone"] = meta.get("zone")

    geographic_concentration = [
        {"county": county, "project_count": data["count"], "zone": data["zone"], "n": data["count"]}
        for county, data in county_data.items()
    ]

    # Sort by project count
    geographic_concentration.sort(key=lambda x: x["project_count"], reverse=True)

    # Keep top 20 counties
    geographic_concentration = geographic_concentration[:20]

    # --- SPECIFIC DEVELOPERS ---
    # Uses KEY_DEVELOPERS constant for configuration
    # Q8 fix: Use word boundary matching to prevent false positives
    corpus_median = corpus_stats["security_per_kw"]["median"]
    specific_developers = {}

    def get_developer_values_by_pattern(pattern: str) -> Tuple[List[float], int]:
        """
        Find all security values for developers matching pattern using word boundaries.

        Uses regex word boundary matching (\\b) instead of naive substring matching
        to prevent false positives (e.g., "AES" matching "CAESAR").

        Args:
            pattern: Developer name pattern (e.g., "RWE", "SAMSUNG")

        Returns:
            Tuple of (security_values_list, total_project_count)

        Raises:
            ValueError: If pattern is shorter than MIN_PATTERN_LENGTH
        """
        # Safety guard: reject short patterns that could cause false positives
        if len(pattern) < MIN_PATTERN_LENGTH:
            raise ValueError(
                f"Pattern '{pattern}' is too short (min {MIN_PATTERN_LENGTH} chars). "
                f"Short patterns risk matching unrelated developer names."
            )

        values = []
        project_count = 0

        # Use word boundary regex for safer matching
        # \b ensures we match whole words, not substrings
        # e.g., "RWE" matches "RWE Solar Development" but not "CRWEST POWER"
        pattern_regex = re.compile(rf'\b{re.escape(pattern)}\b', re.IGNORECASE)

        for dev_name, sec_values in developer_security.items():
            if pattern_regex.search(dev_name):
                values.extend(sec_values)
                project_count += developer_projects.get(dev_name, 0)

        return values, project_count

    for target_dev in KEY_DEVELOPERS:
        values, proj_count = get_developer_values_by_pattern(target_dev)
        if values:
            stats = compute_statistics(values)

            # Calculate vs corpus median
            if corpus_median and stats["median"]:
                pct_diff = ((stats["median"] - corpus_median) / corpus_median) * 100
                vs_corpus = f"+{pct_diff:.1f}%" if pct_diff >= 0 else f"{pct_diff:.1f}%"
                assessment = "ABOVE_MARKET" if pct_diff > 0 else "BELOW_MARKET" if pct_diff < 0 else "AT_MARKET"
            else:
                vs_corpus = "N/A"
                assessment = "INSUFFICIENT_DATA"

            # Q8 fix: Include explicit comparison statement for LLM
            comparison_statement = None
            if corpus_median and stats["median"]:
                if stats["median"] > corpus_median:
                    comparison_statement = f"{target_dev}'s ${stats['median']:.2f}/kW is ABOVE corpus median ${corpus_median:.2f}/kW"
                elif stats["median"] < corpus_median:
                    comparison_statement = f"{target_dev}'s ${stats['median']:.2f}/kW is BELOW corpus median ${corpus_median:.2f}/kW"
                else:
                    comparison_statement = f"{target_dev}'s ${stats['median']:.2f}/kW EQUALS corpus median ${corpus_median:.2f}/kW"

            specific_developers[target_dev] = {
                "project_count": proj_count,
                "projects_with_security_data": stats["n"],
                "median_security_per_kw": stats["median"],
                "mean_security_per_kw": stats["mean"],
                "min_security_per_kw": stats["min"],
                "max_security_per_kw": stats["max"],
                "vs_corpus_median": vs_corpus,
                "assessment": assessment,
                "comparison_statement": comparison_statement,
                "corpus_median_for_reference": corpus_median,
                "n": stats["n"]
            }
        else:
            # Developer not found with fuzzy matching
            pass

    # --- DATA QUALITY FLAGS ---
    data_quality = {
        "low_sample_warnings": [],
        "missing_data_warnings": []
    }

    # Check fuel types
    for fuel_name, data in by_fuel_type.items():
        if data["n"] > 0 and data["n"] < MIN_SAMPLE_SIZE:
            data_quality["low_sample_warnings"].append({
                "category": "fuel_type",
                "bucket": fuel_name,
                "sample_size": data["n"],
                "warning": f"Low sample size (n={data['n']} < {MIN_SAMPLE_SIZE})"
            })

    # Check zones
    for zone_name, data in by_zone.items():
        if data["n"] > 0 and data["n"] < MIN_SAMPLE_SIZE:
            data_quality["low_sample_warnings"].append({
                "category": "zone",
                "bucket": zone_name,
                "sample_size": data["n"],
                "warning": f"Low sample size (n={data['n']} < {MIN_SAMPLE_SIZE})"
            })

    # Check TSPs
    for tsp_data in tsp_rankings:
        if tsp_data["n"] > 0 and tsp_data["n"] < MIN_SAMPLE_SIZE:
            data_quality["low_sample_warnings"].append({
                "category": "tsp",
                "bucket": tsp_data["tsp"],
                "sample_size": tsp_data["n"],
                "warning": f"Low sample size (n={tsp_data['n']} < {MIN_SAMPLE_SIZE})"
            })

    # Check specific developers
    for dev_name, data in specific_developers.items():
        if data["n"] > 0 and data["n"] < MIN_SAMPLE_SIZE:
            data_quality["low_sample_warnings"].append({
                "category": "specific_developer",
                "bucket": dev_name,
                "sample_size": data["n"],
                "warning": f"Low sample size (n={data['n']} < {MIN_SAMPLE_SIZE})"
            })

    # Missing data warnings
    missing_security_pct = ((total_projects - len(projects_with_security)) / total_projects * 100) if total_projects > 0 else 0
    if missing_security_pct > 20:
        data_quality["missing_data_warnings"].append({
            "field": "security_per_kw",
            "missing_count": total_projects - len(projects_with_security),
            "missing_pct": round(missing_security_pct, 1),
            "warning": f"{missing_security_pct:.1f}% of projects missing security_per_kw data"
        })

    # --- ASSEMBLE FINAL ANALYTICS ---
    analytics = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "chromadb_path": chromadb_path,
        "corpus_stats": corpus_stats,
        "by_fuel_type": by_fuel_type,
        "by_zone": by_zone,
        "tsp_rankings": tsp_rankings,
        "developer_analysis": developer_analysis,
        "geographic_concentration": geographic_concentration,
        "specific_developers": specific_developers,
        "data_quality": data_quality
    }

    return analytics


def save_analytics(analytics: Dict[str, Any], output_path: str) -> None:
    """Save analytics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analytics, f, indent=2, ensure_ascii=False)

    print(f"Analytics saved to: {output_path}")


def load_analytics(path: str = DEFAULT_OUTPUT_PATH) -> Optional[Dict[str, Any]]:
    """Load analytics from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        print(f"Error loading analytics: {e}")
        return None


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate corpus analytics from ChromaDB"
    )
    parser.add_argument(
        "--chromadb-path",
        default=DEFAULT_CHROMADB_PATH,
        help=f"Path to ChromaDB (default: {DEFAULT_CHROMADB_PATH})"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for analytics JSON (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print analytics summary to console"
    )

    args = parser.parse_args()

    print(f"Connecting to ChromaDB at: {args.chromadb_path}")
    analytics = generate_corpus_analytics(args.chromadb_path)

    if args.verbose:
        print("\n" + "=" * 60)
        print("CORPUS ANALYTICS SUMMARY")
        print("=" * 60)
        print(f"Total Projects: {analytics['corpus_stats']['total_projects']}")
        print(f"Projects with Security Data: {analytics['corpus_stats']['projects_with_security_data']}")
        print(f"Median $/kW: ${analytics['corpus_stats']['security_per_kw']['median']}")
        print(f"Mean $/kW: ${analytics['corpus_stats']['security_per_kw']['mean']}")
        print(f"\nTop 5 TSPs by avg $/kW:")
        for tsp in analytics["tsp_rankings"][:5]:
            print(f"  {tsp['rank']}. {tsp['tsp']}: ${tsp['avg_security_per_kw']}/kW (n={tsp['n']})")
        print(f"\nMulti-zone developers: {len(analytics['developer_analysis']['multi_zone_developers'])}")
        print(f"Diversified portfolios: {len(analytics['developer_analysis']['diversified_portfolios'])}")

        if analytics["data_quality"]["low_sample_warnings"]:
            print(f"\n{len(analytics['data_quality']['low_sample_warnings'])} low sample size warnings")

        print("=" * 60)

    save_analytics(analytics, args.output)
    print("Done!")


if __name__ == "__main__":
    main()

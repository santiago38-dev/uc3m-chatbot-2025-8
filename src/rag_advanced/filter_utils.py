"""
Filter Utilities - Extract and build filters for comparative queries.

Separated from vector_store.py to avoid circular imports.
This module is imported by both vector_store.py and chain.py.
"""

import re
from typing import Dict, List, Any, Set

from .alias_expander import (
    expand_parent_company_aliases,
    expand_tsp_aliases
)


# =============================================================================
# FUEL TYPE KEYWORDS - Single source of truth
# These are generic technology terms that should NEVER be treated as project names
# Used to prevent false positives in LLM metadata extraction
# =============================================================================

FUEL_TYPE_KEYWORDS: Set[str] = {
    # English
    'battery', 'batteries', 'storage', 'bess', 'energy storage',
    'solar', 'photovoltaic', 'pv', 'sun',
    'wind', 'turbine', 'turbines',
    'gas', 'natural gas', 'ng', 'combined cycle', 'peaker',
    # Spanish
    'bateria', 'baterias', 'almacenamiento',
    'solares', 'fotovoltaico', 'fotovoltaica',
    'viento', 'vientos', 'eolico', 'eolica',
    # Generic terms that are NOT project names
    'project', 'projects', 'proyecto', 'proyectos',
    'plant', 'plants', 'facility', 'facilities',
    'farm', 'farms', 'park', 'parks',
}


def is_generic_fuel_term(term: str) -> bool:
    """
    Check if a term is a generic fuel type or technology keyword.

    These should NOT be treated as project names by LLM metadata extraction.

    Args:
        term: The term to check (e.g., "battery", "solar", "Headcamp")

    Returns:
        True if the term is generic and should be filtered out
    """
    if not term:
        return True

    term_lower = term.lower().strip()

    # Direct match
    if term_lower in FUEL_TYPE_KEYWORDS:
        return True

    # Partial match for compound terms (e.g., "solar project")
    for keyword in FUEL_TYPE_KEYWORDS:
        if term_lower == keyword:
            return True

    # Very short terms are likely generic (e.g., "PV", "WT")
    if len(term_lower) <= 2:
        return True

    return False


def filter_project_names(names: List[str]) -> List[str]:
    """
    Filter out generic fuel type terms from a list of extracted project names.

    This is critical for preventing false positives in the project name warning system.
    The LLM sometimes extracts "battery", "solar" as project names when the user
    asks about "battery and solar projects" - this filters those out.

    Args:
        names: List of extracted "project names" from LLM

    Returns:
        Filtered list with only likely real project names
    """
    if not names:
        return []

    return [name for name in names if not is_generic_fuel_term(name)]


# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================

# Known parent company patterns for extraction
PARENT_PATTERNS = {
    'NEXTERA': r'nextera|next\s*era',
    'RWE': r'\brwe\b',
    'SAMSUNG': r'samsung',
    'INVENERGY': r'invenergy',
    'EDF': r'\bedf\b',
    'ENEL': r'enel',
    'AES': r'\baes\b',
    'ENGIE': r'engie',
    'INTERSECT': r'intersect',
    'HECATE': r'hecate',
    'CLEARWAY': r'clearway',
    'APEX': r'apex',
    'VISTRA': r'vistra',
    'PLUS POWER': r'plus\s*power',
    'KEY CAPTURE': r'key\s*capture',
    'BROAD REACH': r'broad\s*reach',
    'JUPITER': r'jupiter',
    'ORSTED': r'orsted|ørsted',
    'CANADIAN SOLAR': r'canadian\s*solar|recurrent',
    'LIGHTSOURCE BP': r'lightsource|bp\s*solar',
    'ORIGIS': r'origis',
    '8MINUTE': r'8\s*minute',
}

# Known TSP patterns
TSP_PATTERNS = {
    'ONCOR': r'oncor',
    'CENTERPOINT': r'centerpoint|cnp|cpnt',
    'AEP': r'\baep\b|aep\s*texas',
    'TNMP': r'tnmp|texas[\s-]new\s*mexico',
    'ETT': r'\bett\b|electric\s*transmission\s*texas',
    'LCRA': r'lcra|lower\s*colorado',
    'SHARYLAND': r'sharyland',
}


# =============================================================================
# FILTER EXTRACTION
# =============================================================================

def extract_multi_filters_from_query(query: str) -> Dict[str, Any]:
    """
    Extract filters supporting MULTIPLE values for comparative queries.

    This is critical for queries like:
    - "Compare RWE vs SAMSUNG battery projects"  -> parent_company: ['RWE', 'SAMSUNG']
    - "ONCOR vs Centerpoint territories"         -> tsp_normalized: ['ONCOR', 'CENTERPOINT']
    - "Battery vs solar cost comparison"         -> fuel_type: ['OTH', 'SOL']

    Returns dict with:
    - Single values as strings (for backward compatibility where no comparison detected)
    - Multiple values as lists (when comparison detected)
    """
    filters: Dict[str, Any] = {}
    q = query.lower()

    def find_all_matches(patterns: Dict[str, str]) -> List[str]:
        """Find all matching patterns, not just the first."""
        matches = []
        for canonical, pattern in patterns.items():
            if re.search(rf"\b({pattern})\b", q, re.IGNORECASE):
                matches.append(canonical)
        return matches

    # Detect comparison keywords
    is_comparative = bool(re.search(
        r'\bvs\.?\b|\bversus\b|\bcompare\b|\bcomparison\b|\bdifference\b|\bbetween\b',
        q, re.IGNORECASE
    ))

    # === PARENT COMPANY (multi-value support) ===
    parent_matches = find_all_matches(PARENT_PATTERNS)
    if parent_matches:
        if len(parent_matches) == 1 and not is_comparative:
            filters['parent_company'] = parent_matches[0]
        else:
            filters['parent_company'] = parent_matches

    # === TSP (multi-value support) ===
    tsp_matches = find_all_matches(TSP_PATTERNS)
    if tsp_matches:
        if len(tsp_matches) == 1 and not is_comparative:
            filters['tsp_normalized'] = tsp_matches[0]
        else:
            filters['tsp_normalized'] = tsp_matches

    # === FUEL TYPE (multi-value support for "battery vs solar") ===
    fuel_matches = []
    if re.search(r'\b(battery|bateria|baterias|storage|bess)\b', q, re.IGNORECASE):
        fuel_matches.append('OTH')
    if re.search(r'\b(solar|solares|pv|sun)\b', q, re.IGNORECASE):
        fuel_matches.append('SOL')
    if re.search(r'\b(wind|viento|vientos)\b', q, re.IGNORECASE):
        fuel_matches.append('WIN')
    if re.search(r'\b(gas|natural\s*gas)\b', q, re.IGNORECASE):
        fuel_matches.append('GAS')

    if fuel_matches:
        if len(fuel_matches) == 1 and not is_comparative:
            filters['fuel_type'] = fuel_matches[0]
        else:
            filters['fuel_type'] = fuel_matches

    # === ZONE (usually single, but support multi for completeness) ===
    zone_matches = []
    if re.search(r'\b(coast|coastal|houston)\b', q, re.IGNORECASE):
        zone_matches.append('COAST')
    if re.search(r'\b(west|western)\b', q, re.IGNORECASE):
        zone_matches.append('WEST')
    if re.search(r'\b(north|northern)\b', q, re.IGNORECASE):
        zone_matches.append('NORTH')
    if re.search(r'\b(south|southern)\b', q, re.IGNORECASE):
        zone_matches.append('SOUTH')
    if re.search(r'\bpanhandle\b', q, re.IGNORECASE):
        zone_matches.append('PANHANDLE')

    if zone_matches:
        if len(zone_matches) == 1:
            filters['zone'] = zone_matches[0]
        else:
            filters['zone'] = zone_matches

    # === INR (Interconnection Request Number) ===
    # Pattern: YYINR####  e.g., "25INR0138", "24INR0485"
    inr_match = re.search(r'\b(\d{2}INR\d{4})\b', query, re.IGNORECASE)
    if inr_match:
        filters['inr'] = inr_match.group(1).upper()

    # === NUMERIC FILTERS ===
    # Security per kW threshold (e.g., ">$100/kW", "over $100 per kW")
    sec_match = re.search(
        r'(?:>|over|above|more\s+than|exceeding)\s*\$?\s*(\d+)\s*(?:/|per)\s*kw',
        q, re.IGNORECASE
    )
    if sec_match:
        filters['security_per_kw_min'] = float(sec_match.group(1))

    # Capacity threshold (e.g., ">100 MW", "over 100 MW")
    cap_match = re.search(
        r'(?:>|over|above|more\s+than)\s*(\d+)\s*(?:mw|megawatt)',
        q, re.IGNORECASE
    )
    if cap_match:
        filters['capacity_mw_min'] = float(cap_match.group(1))

    return filters


def build_chromadb_where_clause(filters: Dict[str, Any], expand_aliases: bool = True) -> Dict:
    """
    Build a ChromaDB $where clause from extracted filters.

    Handles:
    - Single values: {"field": {"$eq": value}}
    - Multiple values: {"field": {"$in": [values]}} with alias expansion
    - Numeric thresholds: {"field": {"$gte": value}}

    Args:
        filters: Dict from extract_multi_filters_from_query()
        expand_aliases: Whether to expand parent_company/tsp aliases

    Returns:
        ChromaDB-compatible where clause
    """
    if not filters:
        return {}

    conditions = []

    for key, value in filters.items():
        if key == 'security_per_kw_min':
            # Numeric >= filter
            conditions.append({'security_per_kw': {'$gte': value}})

        elif key == 'capacity_mw_min':
            # Numeric >= filter
            conditions.append({'capacity_mw': {'$gte': value}})

        elif key == 'parent_company':
            if isinstance(value, list):
                # Multi-value: expand aliases and use $in
                if expand_aliases:
                    expanded = expand_parent_company_aliases(value)
                else:
                    expanded = value
                conditions.append({'parent_company': {'$in': expanded}})
            else:
                # Single value: still expand for exact match
                if expand_aliases:
                    expanded = expand_parent_company_aliases([value])
                    if len(expanded) > 1:
                        conditions.append({'parent_company': {'$in': expanded}})
                    else:
                        conditions.append({'parent_company': {'$eq': expanded[0]}})
                else:
                    conditions.append({'parent_company': {'$eq': value}})

        elif key == 'tsp_normalized':
            if isinstance(value, list):
                if expand_aliases:
                    expanded = expand_tsp_aliases(value)
                else:
                    expanded = value
                conditions.append({'tsp_normalized': {'$in': expanded}})
            else:
                if expand_aliases:
                    expanded = expand_tsp_aliases([value])
                    if len(expanded) > 1:
                        conditions.append({'tsp_normalized': {'$in': expanded}})
                    else:
                        conditions.append({'tsp_normalized': {'$eq': expanded[0]}})
                else:
                    conditions.append({'tsp_normalized': {'$eq': value}})

        elif isinstance(value, list):
            # Generic multi-value (fuel_type, zone, etc.)
            conditions.append({key: {'$in': value}})

        else:
            # Single value filter
            conditions.append({key: {'$eq': value}})

    # Combine conditions
    if not conditions:
        return {}
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {'$and': conditions}

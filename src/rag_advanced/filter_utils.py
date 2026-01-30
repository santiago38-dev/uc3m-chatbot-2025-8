"""
Filter Utilities - Extract and build filters for comparative queries.

Separated from vector_store.py to avoid circular imports.
This module is imported by both vector_store.py and chain.py.
"""

import re
from typing import Dict, List, Any, Optional, Tuple

from .alias_expander import (
    expand_parent_company_aliases,
    expand_tsp_aliases
)


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

# Common Texas county patterns (most common for ERCOT projects)
COUNTY_PATTERNS = [
    r'travis\s*county',
    r'brazoria\s*county',
    r'hidalgo\s*county',
    r'pecos\s*county',
    r'reeves\s*county',
    r'culberson\s*county',
    r'wharton\s*county',
    r'matagorda\s*county',
    r'bee\s*county',
    r'nueces\s*county',
    r'cameron\s*county',
    r'willacy\s*county',
    r'starr\s*county',
    r'webb\s*county',
    r'zapata\s*county',
    r'kenedy\s*county',
    r'kleberg\s*county',
    r'jim\s*wells\s*county',
    r'live\s*oak\s*county',
    r'mcmullen\s*county',
    r'la\s*salle\s*county',
    r'dimmit\s*county',
    r'maverick\s*county',
    r'zavala\s*county',
    r'frio\s*county',
    r'atascosa\s*county',
    r'wilson\s*county',
    r'karnes\s*county',
    r'goliad\s*county',
    r'victoria\s*county',
    r'calhoun\s*county',
    r'jackson\s*county',
    r'lavaca\s*county',
    r'dewitt\s*county',
    r'gonzales\s*county',
    r'guadalupe\s*county',
    r'comal\s*county',
    r'hays\s*county',
    r'caldwell\s*county',
    r'bastrop\s*county',
    r'lee\s*county',
    r'burleson\s*county',
    r'brazos\s*county',
    r'robertson\s*county',
    r'milam\s*county',
    r'williamson\s*county',
    r'bell\s*county',
    r'falls\s*county',
    r'mclennan\s*county',
    r'coryell\s*county',
    r'hamilton\s*county',
    r'lampasas\s*county',
    r'burnet\s*county',
    r'llano\s*county',
    r'mason\s*county',
    r'kimble\s*county',
    r'kerr\s*county',
    r'bandera\s*county',
    r'medina\s*county',
    r'bexar\s*county',
    r'kendall\s*county',
    r'blanco\s*county',
    r'gillespie\s*county',
    r'real\s*county',
    r'uvalde\s*county',
    r'kinney\s*county',
    r'val\s*verde\s*county',
    r'edwards\s*county',
    r'sutton\s*county',
    r'schleicher\s*county',
    r'menard\s*county',
    r'mcculloch\s*county',
    r'san\s*saba\s*county',
    r'mills\s*county',
    r'brown\s*county',
    r'coleman\s*county',
    r'runnels\s*county',
    r'tom\s*green\s*county',
    r'concho\s*county',
    r'irion\s*county',
    r'crockett\s*county',
    r'terrell\s*county',
    r'brewster\s*county',
    r'presidio\s*county',
    r'jeff\s*davis\s*county',
    r'hudspeth\s*county',
    r'el\s*paso\s*county',
]

# Project name suffixes for extraction
PROJECT_SUFFIXES = [
    'BESS', 'Battery', 'Storage', 'Solar', 'Wind', 'Energy Storage',
    'Energy Storage Plant', 'Plant', 'Farm', 'Project', 'Generation',
    'Power', 'Facility', 'Station'
]


# =============================================================================
# PROJECT NAME EXTRACTION
# =============================================================================

def extract_single_project_name(query: str) -> Optional[str]:
    """
    Extract a single project name from a query (non-comparison).

    Examples:
        "What is the security deposit for Champaign BESS?" -> "Champaign BESS"
        "Tell me about Quantum Storage" -> "Quantum Storage"
        "Mustang Solar project details" -> "Mustang Solar"
        "security for 'Headcamp Energy Storage Plant'" -> "Headcamp Energy Storage Plant"

    Returns:
        Project name if found, None otherwise
    """
    # Pattern 1: Quoted project name (single or double quotes)
    quoted = re.search(r'["\']([^"\']+(?:BESS|Storage|Solar|Wind|Plant|Farm|Project|Energy))["\']', query, re.IGNORECASE)
    if quoted:
        return quoted.group(1).strip()

    # Pattern 2: "for X BESS/Storage/Solar/etc" - most common pattern
    # e.g., "security deposit for Champaign BESS"
    for_pattern = re.search(
        r'\bfor\s+([A-Z][A-Za-z0-9\s]+?(?:BESS|Battery|Storage|Solar|Wind|Energy Storage Plant|Energy Storage|Plant|Farm|Project|Generation|Power|Facility|Station))\b',
        query
    )
    if for_pattern:
        return for_pattern.group(1).strip()

    # Pattern 3: "about X BESS/Storage/Solar/etc"
    about_pattern = re.search(
        r'\babout\s+(?:the\s+)?([A-Z][A-Za-z0-9\s]+?(?:BESS|Battery|Storage|Solar|Wind|Energy Storage Plant|Energy Storage|Plant|Farm|Project|Generation|Power|Facility|Station))\b',
        query
    )
    if about_pattern:
        return about_pattern.group(1).strip()

    # Pattern 4: "X BESS/Storage terms" or "X Storage's security"
    suffix_pattern = re.search(
        r'\b([A-Z][A-Za-z0-9\s]+?(?:BESS|Battery|Storage|Solar|Wind|Energy Storage Plant|Energy Storage|Plant|Farm|Project|Generation|Power|Facility|Station))(?:\'s|\s+terms|\s+security|\s+deposit|\s+agreement|\s+contract|\s+details)',
        query
    )
    if suffix_pattern:
        return suffix_pattern.group(1).strip()

    # Pattern 5: Just "X BESS" or "X Storage" at start of phrase
    standalone = re.search(
        r'\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\s+(?:BESS|Battery|Storage|Solar|Wind|Energy Storage Plant|Energy Storage|Plant|Farm|Project))\b',
        query
    )
    if standalone:
        return standalone.group(1).strip()

    return None


def extract_project_names_from_comparison(query: str) -> Optional[List[str]]:
    """
    Extract project names from a comparison query.

    Examples:
        "Compare Headcamp Energy Storage Plant to Quantum Storage"
            -> ['Headcamp Energy Storage Plant', 'Quantum Storage']
        "Headcamp vs Quantum - what are the differences?"
            -> ['Headcamp', 'Quantum']
        "Compare project A to project B"
            -> ['project A', 'project B']

    Returns:
        List of project names if found, None otherwise
    """
    # Pattern 1: "Compare X to Y" or "Compare X and Y"
    pattern1 = re.search(
        r'compare\s+(.+?)\s+(?:to|and|with|versus|vs\.?)\s+(.+?)(?:\s*[-–—]|\s*\?|$)',
        query, re.IGNORECASE
    )
    if pattern1:
        name1 = pattern1.group(1).strip()
        name2 = pattern1.group(2).strip()
        # Clean up common suffixes
        name2 = re.sub(r'\s*[-–—].*$', '', name2).strip()
        name2 = re.sub(r'\s*what\s+are.*$', '', name2, flags=re.IGNORECASE).strip()
        if name1 and name2:
            return [name1, name2]

    # Pattern 2: "X vs Y" or "X versus Y"
    pattern2 = re.search(
        r'([A-Z][A-Za-z\s]+(?:Storage|Plant|Solar|Wind|BESS|Energy|Farm|Project)?)\s+(?:vs\.?|versus)\s+([A-Z][A-Za-z\s]+(?:Storage|Plant|Solar|Wind|BESS|Energy|Farm|Project)?)',
        query
    )
    if pattern2:
        name1 = pattern2.group(1).strip()
        name2 = pattern2.group(2).strip()
        if name1 and name2:
            return [name1, name2]

    return None


def normalize_project_name_for_search(name: str) -> List[str]:
    """
    Generate variations of a project name for fuzzy matching.

    "Champaign BESS" ->
        ["Champaign BESS", "Champaign", "Champaign Battery", "Champaign Storage",
         "Champaign Energy Storage", "Champaign Battery Energy Storage"]

    "Headcamp Energy Storage Plant" ->
        ["Headcamp Energy Storage Plant", "Headcamp", "Headcamp Energy", "Headcamp Storage"]
    """
    variations = [name]
    seen = {name.lower()}  # Track lowercase to avoid duplicates

    def add_variation(v: str):
        v = v.strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            variations.append(v)

    # Add first word as variation (often the unique identifier)
    words = name.split()
    if len(words) > 1:
        add_variation(words[0])
        # Add first two words
        if len(words) > 2:
            add_variation(' '.join(words[:2]))

    # Remove common suffixes and add base as variation
    common_suffixes = [
        'Battery Energy Storage System', 'Energy Storage System', 'Energy Storage Plant',
        'Battery Energy Storage', 'Storage Plant', 'Storage System', 'Energy Storage',
        'Storage', 'Plant', 'Solar', 'Wind', 'BESS', 'Farm', 'Project', 'Energy', 'Battery'
    ]
    base_name = None
    for suffix in common_suffixes:
        if name.lower().endswith(suffix.lower()) and name.lower() != suffix.lower():
            base = name[:-len(suffix)].strip()
            if base:
                add_variation(base)
                if base_name is None:
                    base_name = base

    # If we found a base name (e.g., "Champaign" from "Champaign BESS"),
    # generate common suffix variations
    if base_name:
        storage_suffixes = [
            'BESS', 'Battery', 'Storage', 'Energy Storage', 'Battery Storage',
            'Battery Energy Storage', 'Energy Storage System'
        ]
        for suffix in storage_suffixes:
            add_variation(f"{base_name} {suffix}")

    # Also handle case where name might be partial (just "Champaign")
    # Add common storage suffixes
    if len(words) == 1:
        for suffix in ['BESS', 'Battery', 'Storage', 'Energy Storage', 'Solar', 'Wind']:
            add_variation(f"{name} {suffix}")

    return variations


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

    # === PROJECT NAME ===
    # For comparative queries, extract multiple project names
    # For single-project queries, extract the project name for hard filtering
    if is_comparative:
        project_names = extract_project_names_from_comparison(query)
        if project_names:
            filters['project_name'] = project_names  # List for $in clause
    else:
        single_project = extract_single_project_name(query)
        if single_project:
            filters['project_name'] = single_project  # String for $eq clause

    # === COUNTY ===
    # Extract county name for location-based queries
    for pattern in COUNTY_PATTERNS:
        county_match = re.search(pattern, q, re.IGNORECASE)
        if county_match:
            # Extract just the county name (capitalize properly)
            matched = county_match.group(0)
            # Remove "county" suffix and capitalize
            county_name = re.sub(r'\s*county$', '', matched, flags=re.IGNORECASE).strip().title()
            filters['county'] = county_name
            break  # Only take first match

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

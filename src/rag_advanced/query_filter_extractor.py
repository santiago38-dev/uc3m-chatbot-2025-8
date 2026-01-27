"""
Query Filter Extractor - Extract ChromaDB filters from natural language queries
================================================================================

Source of Truth:
- Zones: COUNTY_ZONES from src/chunks/metadata.py
- Developers: PARENT_MAPPING from src/chunks/metadata.py
- Fuel types: WIN, SOL, OTH, GAS (from ERCOT CSV)
- TSPs: Normalized names from indexing

Handles:
1. Numeric comparisons: "> $100/kW", "above 50", "under $50/kW"
2. Zone filters: "West Texas", "coastal", "panhandle area"
3. Fuel type filters: "battery storage", "solar projects", "wind farms"
4. TSP filters: "ONCOR", "Centerpoint", "ETT"
5. Developer filters: "RWE projects", "Samsung battery"
6. COMPARATIVE QUERIES: "X vs Y", "compare A and B" -> extract BOTH entities

Returns ChromaDB-compatible where clauses for hard pre-filtering.
"""

import re
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field

# Import source of truth
from src.chunks.metadata import COUNTY_ZONES, PARENT_MAPPING


@dataclass
class NumericFilter:
    """Represents a numeric comparison filter."""
    field: str
    operator: str  # $gt, $gte, $lt, $lte, $eq
    value: float


@dataclass
class ExtractedFilters:
    """Container for all extracted filters - supports multi-value for comparisons."""
    # field -> single value OR list of values for comparisons
    equality_filters: Dict[str, Union[str, List[str]]] = field(default_factory=dict)
    numeric_filters: List[NumericFilter] = field(default_factory=list)
    is_comparative: bool = False  # Flag for comparative queries

    def to_chromadb_where(self) -> Optional[Dict]:
        """Convert to ChromaDB where clause format.

        Uses $in operator for multi-value filters (comparative queries).
        """
        conditions = []

        # Add equality filters
        for fld, value in self.equality_filters.items():
            if isinstance(value, list):
                # Multi-value: use $in operator
                conditions.append({fld: {"$in": value}})
            else:
                # Single value: use $eq
                conditions.append({fld: {"$eq": value}})

        # Add numeric filters
        for nf in self.numeric_filters:
            conditions.append({nf.field: {nf.operator: nf.value}})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    def is_empty(self) -> bool:
        return not self.equality_filters and not self.numeric_filters


# =============================================================================
# COMPARATIVE QUERY DETECTION
# =============================================================================

COMPARATIVE_PATTERNS = [
    r'\bcompare\b',
    r'\bcomparison\b',
    r'\bvs\.?\b',
    r'\bversus\b',
    r'\band\b.*\bprojects?\b',  # "RWE and SAMSUNG projects"
    r'\bbetween\b',
    r'\bdifference\s+between\b',
]


def is_comparative_query(query: str) -> bool:
    """Detect if query is comparing multiple entities."""
    q_lower = query.lower()
    return any(re.search(p, q_lower) for p in COMPARATIVE_PATTERNS)


# =============================================================================
# ZONE DETECTION - Uses COUNTY_ZONES as source of truth
# =============================================================================

VALID_ZONES: Set[str] = set(COUNTY_ZONES.values())

# Zone keyword mappings - ORDER MATTERS: specific zones first, then general
ZONE_KEYWORDS = [
    ("PANHANDLE", ["panhandle", "amarillo", "lubbock"]),
    ("COAST", ["coast", "coastal", "houston", "gulf", "galveston", "brazoria", "corpus christi"]),
    ("NORTH", ["north texas", "north tx", "dallas", "dfw", "fort worth", "tarrant"]),
    ("SOUTH", ["south texas", "south tx", "san antonio", "rio grande", "laredo", "valley"]),
    ("CENTRAL", ["central texas", "central tx", "austin", "waco", "temple"]),
    ("WEST", ["west texas", "west tx", "western texas", "permian", "midland", "odessa", "pecos"]),
]


def extract_zones(query: str) -> List[str]:
    """Extract ALL matching zones from query (for comparative queries)."""
    q_lower = query.lower()
    found = []

    for zone, keywords in ZONE_KEYWORDS:
        for kw in keywords:
            if kw in q_lower and zone not in found:
                found.append(zone)
                break  # Only add each zone once

    return found


def extract_zone(query: str) -> Optional[str]:
    """Extract single zone (first match) - backwards compatibility."""
    zones = extract_zones(query)
    return zones[0] if zones else None


# =============================================================================
# FUEL TYPE DETECTION
# =============================================================================

FUEL_TYPE_KEYWORDS = {
    "OTH": ["battery", "bess", "storage", "energy storage", "battery storage"],
    "SOL": ["solar", "pv", "photovoltaic", "solar farm", "solar project"],
    "WIN": ["wind", "wind farm", "wind project", "wind turbine"],
    "GAS": ["gas", "natural gas", "ccgt", "combustion turbine", "peaker", "gas turbine"],
}


def extract_fuel_types(query: str) -> List[str]:
    """Extract ALL matching fuel types from query (for comparative queries)."""
    q_lower = query.lower()
    found = []

    for fuel_type, keywords in FUEL_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower and fuel_type not in found:
                found.append(fuel_type)
                break

    return found


def extract_fuel_type(query: str) -> Optional[str]:
    """Extract single fuel type - backwards compatibility."""
    types = extract_fuel_types(query)
    return types[0] if types else None


# =============================================================================
# TSP DETECTION
# =============================================================================

TSP_KEYWORDS = {
    "ONCOR": ["oncor"],
    "CENTERPOINT": ["centerpoint", "center point", "cnp"],
    "AEP TEXAS": ["aep", "aep texas"],
    "TNMP": ["tnmp", "texas-new mexico"],
    "ETT": ["ett", "electric transmission texas"],
    "LCRA TSC": ["lcra", "lower colorado"],
}


def extract_tsps(query: str) -> List[str]:
    """Extract ALL matching TSPs from query (for comparative queries)."""
    q_lower = query.lower()
    found = []

    for tsp, keywords in TSP_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf'\b{re.escape(kw)}\b', q_lower) and tsp not in found:
                found.append(tsp)
                break

    return found


def extract_tsp(query: str) -> Optional[str]:
    """Extract single TSP - backwards compatibility."""
    tsps = extract_tsps(query)
    return tsps[0] if tsps else None


# =============================================================================
# DEVELOPER DETECTION - Uses PARENT_MAPPING as source of truth
# =============================================================================

def extract_developers(query: str) -> List[str]:
    """Extract ALL matching developers from query (for comparative queries)."""
    q_upper = query.upper()
    found = []

    # Check for parent company names directly
    for parent in PARENT_MAPPING.keys():
        if parent in q_upper and parent not in found:
            found.append(parent)

    # Check for aliases
    for parent, aliases in PARENT_MAPPING.items():
        if parent not in found:
            for alias in aliases:
                if alias.upper() in q_upper:
                    found.append(parent)
                    break

    return found


def extract_developer(query: str) -> Optional[str]:
    """Extract single developer - backwards compatibility."""
    devs = extract_developers(query)
    return devs[0] if devs else None


# =============================================================================
# NUMERIC FILTER DETECTION
# =============================================================================

NUMERIC_PATTERNS = [
    (r'(?:>|greater\s+than|above|more\s+than|over|exceeds?)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:/kw|per\s*kw)?',
     '$gt', 'security_per_kw'),
    (r'(?:>=|at\s+least)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:/kw|per\s*kw)?',
     '$gte', 'security_per_kw'),
    (r'(?:<|less\s+than|under|below)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:/kw|per\s*kw)?',
     '$lt', 'security_per_kw'),
    (r'(?:<=|at\s+most|up\s+to)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:/kw|per\s*kw)?',
     '$lte', 'security_per_kw'),
    (r'(?:>|greater\s+than|above|more\s+than|over)\s*(\d+(?:\.\d+)?)\s*(?:mw|megawatt)',
     '$gt', 'capacity_mw'),
    (r'(?:<|less\s+than|under|below)\s*(\d+(?:\.\d+)?)\s*(?:mw|megawatt)',
     '$lt', 'capacity_mw'),
]


def extract_numeric_filters(query: str) -> List[NumericFilter]:
    """Extract numeric comparison filters from query."""
    filters = []
    q_lower = query.lower()

    for pattern, operator, fld in NUMERIC_PATTERNS:
        match = re.search(pattern, q_lower, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            filters.append(NumericFilter(field=fld, operator=operator, value=value))

    return filters


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_query_filters(query: str) -> ExtractedFilters:
    """
    Extract all filters from a natural language query.

    For comparative queries ("X vs Y", "compare A and B"), extracts MULTIPLE
    entities and uses $in operator for ChromaDB filtering.

    Args:
        query: Natural language question

    Returns:
        ExtractedFilters containing equality and numeric filters

    Examples:
        "battery storage in West Texas"
            -> zone=WEST, fuel_type=OTH

        "Compare RWE and SAMSUNG battery projects"
            -> parent_company IN [RWE, SAMSUNG], fuel_type=OTH

        "ONCOR vs Centerpoint force majeure"
            -> tsp_normalized IN [ONCOR, CENTERPOINT]

        "battery vs solar security costs"
            -> fuel_type IN [OTH, SOL]
    """
    comparative = is_comparative_query(query)
    equality_filters: Dict[str, Union[str, List[str]]] = {}

    # Extract zones
    zones = extract_zones(query)
    if zones:
        equality_filters['zone'] = zones if (comparative and len(zones) > 1) else zones[0]

    # Extract fuel types - for comparatives, include all mentioned
    fuel_types = extract_fuel_types(query)
    if fuel_types:
        equality_filters['fuel_type'] = fuel_types if (comparative and len(fuel_types) > 1) else fuel_types[0]

    # Extract TSPs - for comparatives, include all mentioned
    tsps = extract_tsps(query)
    if tsps:
        equality_filters['tsp_normalized'] = tsps if (comparative and len(tsps) > 1) else tsps[0]

    # Extract developers - for comparatives, include all mentioned
    developers = extract_developers(query)
    if developers:
        equality_filters['parent_company'] = developers if (comparative and len(developers) > 1) else developers[0]

    # Extract numeric filters
    numeric_filters = extract_numeric_filters(query)

    return ExtractedFilters(
        equality_filters=equality_filters,
        numeric_filters=numeric_filters,
        is_comparative=comparative
    )


# =============================================================================
# POST-RETRIEVAL VALIDATION
# =============================================================================

def validate_retrieved_docs(
    docs: List,
    filters: ExtractedFilters,
    logger=None
) -> Tuple[List, List[str]]:
    """
    Post-retrieval validation: Check if retrieved docs match extracted filters.

    For multi-value filters, doc passes if it matches ANY value in the list.
    """
    if filters.is_empty():
        return docs, []

    valid_docs = []
    warnings = []

    for doc in docs:
        meta = doc.metadata if hasattr(doc, 'metadata') else {}
        is_valid = True

        # Check equality filters
        for fld, expected in filters.equality_filters.items():
            actual = meta.get(fld)
            if actual:
                if isinstance(expected, list):
                    # Multi-value: pass if actual matches ANY expected value
                    if actual not in expected:
                        warnings.append(
                            f"Doc {meta.get('project_name', 'unknown')}: "
                            f"{fld}={actual} (expected one of {expected})"
                        )
                        is_valid = False
                        break
                else:
                    # Single value
                    if actual != expected:
                        warnings.append(
                            f"Doc {meta.get('project_name', 'unknown')}: "
                            f"{fld}={actual} (expected {expected})"
                        )
                        is_valid = False
                        break

        # Check numeric filters
        if is_valid:
            for nf in filters.numeric_filters:
                actual = meta.get(nf.field)
                if actual is not None:
                    try:
                        actual_val = float(actual)
                        passed = False
                        if nf.operator == '$gt':
                            passed = actual_val > nf.value
                        elif nf.operator == '$gte':
                            passed = actual_val >= nf.value
                        elif nf.operator == '$lt':
                            passed = actual_val < nf.value
                        elif nf.operator == '$lte':
                            passed = actual_val <= nf.value

                        if not passed:
                            warnings.append(
                                f"Doc {meta.get('project_name', 'unknown')}: "
                                f"{nf.field}={actual_val} (expected {nf.operator} {nf.value})"
                            )
                            is_valid = False
                            break
                    except (ValueError, TypeError):
                        pass

        if is_valid:
            valid_docs.append(doc)

    # Log warnings
    if logger and warnings:
        for w in warnings[:5]:
            logger.warning(f"Post-validation filter leak: {w}")
        if len(warnings) > 5:
            logger.warning(f"... and {len(warnings) - 5} more filter leaks")

    return valid_docs, warnings


# =============================================================================
# FORMATTING
# =============================================================================

def format_filters_for_logging(filters: ExtractedFilters) -> str:
    """Format extracted filters for logging/debugging."""
    parts = []

    if filters.is_comparative:
        parts.append("[COMPARATIVE]")

    for fld, value in filters.equality_filters.items():
        if isinstance(value, list):
            parts.append(f"{fld} IN {value}")
        else:
            parts.append(f"{fld}={value}")

    for nf in filters.numeric_filters:
        op_map = {
            '$gt': '>',
            '$gte': '>=',
            '$lt': '<',
            '$lte': '<=',
            '$eq': '='
        }
        op_str = op_map.get(nf.operator, nf.operator)
        parts.append(f"{nf.field} {op_str} {nf.value}")

    return ", ".join(parts) if parts else "(no filters)"

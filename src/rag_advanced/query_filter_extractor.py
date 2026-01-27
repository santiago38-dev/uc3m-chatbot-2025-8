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

Returns ChromaDB-compatible where clauses for hard pre-filtering.
"""

import re
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass

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
    """Container for all extracted filters."""
    equality_filters: Dict[str, str]  # field -> value (exact match)
    numeric_filters: List[NumericFilter]  # numeric comparisons

    def to_chromadb_where(self) -> Optional[Dict]:
        """Convert to ChromaDB where clause format."""
        conditions = []

        # Add equality filters
        for field, value in self.equality_filters.items():
            conditions.append({field: {"$eq": value}})

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
# ZONE DETECTION - Uses COUNTY_ZONES as source of truth
# =============================================================================

# Valid zones from COUNTY_ZONES (deduplicated)
VALID_ZONES: Set[str] = set(COUNTY_ZONES.values())  # {'WEST', 'PANHANDLE', 'COAST', 'NORTH', 'SOUTH', 'CENTRAL'}

# Zone keyword mappings - ORDER MATTERS: specific zones first, then general
# PANHANDLE is checked before WEST because Panhandle is geographically in "West Texas"
ZONE_KEYWORDS = [
    # Most specific first (cities/regions that are unambiguous)
    ("PANHANDLE", ["panhandle", "amarillo", "lubbock"]),
    ("COAST", ["coast", "coastal", "houston", "gulf", "galveston", "brazoria", "corpus christi"]),
    ("NORTH", ["north texas", "north tx", "dallas", "dfw", "fort worth", "tarrant"]),
    ("SOUTH", ["south texas", "south tx", "san antonio", "rio grande", "laredo", "valley"]),
    ("CENTRAL", ["central texas", "central tx", "austin", "waco", "temple"]),
    # WEST is last because "west texas" is a broad term that could include Panhandle
    ("WEST", ["west texas", "west tx", "western texas", "permian", "midland", "odessa", "pecos"]),
]


def extract_zone(query: str) -> Optional[str]:
    """
    Extract zone from query using keyword matching.

    Priority: More specific zones (PANHANDLE, COAST) checked before general (WEST).
    This prevents "West Texas near Amarillo" from matching WEST instead of PANHANDLE.
    """
    q_lower = query.lower()

    # Check in priority order (specific to general)
    for zone, keywords in ZONE_KEYWORDS:
        for kw in keywords:
            if kw in q_lower:
                return zone

    return None


# =============================================================================
# FUEL TYPE DETECTION
# =============================================================================

# Fuel type codes from ERCOT
FUEL_TYPE_KEYWORDS = {
    "OTH": ["battery", "bess", "storage", "energy storage", "battery storage"],
    "SOL": ["solar", "pv", "photovoltaic", "solar farm", "solar project"],
    "WIN": ["wind", "wind farm", "wind project", "wind turbine"],
    "GAS": ["gas", "natural gas", "ccgt", "combustion turbine", "peaker", "gas turbine"],
}


def extract_fuel_type(query: str) -> Optional[str]:
    """Extract fuel type from query using keyword matching."""
    q_lower = query.lower()

    for fuel_type, keywords in FUEL_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                return fuel_type

    return None


# =============================================================================
# TSP DETECTION
# =============================================================================

# TSP normalized names (must match ChromaDB field tsp_normalized)
TSP_KEYWORDS = {
    "ONCOR": ["oncor"],
    "CENTERPOINT": ["centerpoint", "center point", "cnp"],
    "AEP TEXAS": ["aep", "aep texas"],
    "TNMP": ["tnmp", "texas-new mexico"],
    "ETT": ["ett", "electric transmission texas"],
    "LCRA TSC": ["lcra", "lower colorado"],
}


def extract_tsp(query: str) -> Optional[str]:
    """Extract TSP from query using keyword matching."""
    q_lower = query.lower()

    for tsp, keywords in TSP_KEYWORDS.items():
        for kw in keywords:
            # Use word boundary to avoid partial matches
            if re.search(rf'\b{re.escape(kw)}\b', q_lower):
                return tsp

    return None


# =============================================================================
# DEVELOPER DETECTION - Uses PARENT_MAPPING as source of truth
# =============================================================================

def extract_developer(query: str) -> Optional[str]:
    """
    Extract developer/parent company from query.

    Uses PARENT_MAPPING from metadata.py as source of truth.
    """
    q_upper = query.upper()

    # Check for parent company names directly
    for parent in PARENT_MAPPING.keys():
        if parent in q_upper:
            return parent

    # Check for aliases
    for parent, aliases in PARENT_MAPPING.items():
        for alias in aliases:
            if alias.upper() in q_upper:
                return parent

    return None


# =============================================================================
# NUMERIC FILTER DETECTION
# =============================================================================

# Numeric field patterns
NUMERIC_PATTERNS = [
    # Security per kW patterns
    (r'(?:>|greater\s+than|above|more\s+than|over|exceeds?)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:/kw|per\s*kw)?',
     '$gt', 'security_per_kw'),
    (r'(?:>=|at\s+least)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:/kw|per\s*kw)?',
     '$gte', 'security_per_kw'),
    (r'(?:<|less\s+than|under|below)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:/kw|per\s*kw)?',
     '$lt', 'security_per_kw'),
    (r'(?:<=|at\s+most|up\s+to)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:/kw|per\s*kw)?',
     '$lte', 'security_per_kw'),
    # Capacity patterns
    (r'(?:>|greater\s+than|above|more\s+than|over)\s*(\d+(?:\.\d+)?)\s*(?:mw|megawatt)',
     '$gt', 'capacity_mw'),
    (r'(?:<|less\s+than|under|below)\s*(\d+(?:\.\d+)?)\s*(?:mw|megawatt)',
     '$lt', 'capacity_mw'),
]


def extract_numeric_filters(query: str) -> List[NumericFilter]:
    """Extract numeric comparison filters from query."""
    filters = []
    q_lower = query.lower()

    for pattern, operator, field in NUMERIC_PATTERNS:
        match = re.search(pattern, q_lower, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            filters.append(NumericFilter(field=field, operator=operator, value=value))

    return filters


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_query_filters(query: str) -> ExtractedFilters:
    """
    Extract all filters from a natural language query.

    Args:
        query: Natural language question

    Returns:
        ExtractedFilters containing equality and numeric filters

    Examples:
        "battery storage in West Texas"
            -> zone=WEST, fuel_type=OTH

        "security above $100/kW"
            -> security_per_kw > 100

        "ONCOR solar projects under $50/kW"
            -> tsp_normalized=ONCOR, fuel_type=SOL, security_per_kw < 50

        "wind farms in the Panhandle"
            -> zone=PANHANDLE, fuel_type=WIN
    """
    equality_filters = {}

    # Extract zone (priority order handles ambiguity)
    zone = extract_zone(query)
    if zone:
        equality_filters['zone'] = zone

    # Extract fuel type
    fuel_type = extract_fuel_type(query)
    if fuel_type:
        equality_filters['fuel_type'] = fuel_type

    # Extract TSP
    tsp = extract_tsp(query)
    if tsp:
        equality_filters['tsp_normalized'] = tsp

    # Extract developer
    developer = extract_developer(query)
    if developer:
        equality_filters['parent_company'] = developer

    # Extract numeric filters
    numeric_filters = extract_numeric_filters(query)

    return ExtractedFilters(
        equality_filters=equality_filters,
        numeric_filters=numeric_filters
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

    This is a belt-and-suspenders check after ChromaDB filtering.
    Logs warnings for any docs that slipped through.

    Args:
        docs: Retrieved documents
        filters: Extracted filters that should have been applied
        logger: Optional logger for warnings

    Returns:
        Tuple of (valid_docs, warnings)
    """
    if filters.is_empty():
        return docs, []

    valid_docs = []
    warnings = []

    for doc in docs:
        meta = doc.metadata if hasattr(doc, 'metadata') else {}
        is_valid = True

        # Check equality filters
        for field, expected in filters.equality_filters.items():
            actual = meta.get(field)
            if actual and actual != expected:
                warnings.append(
                    f"Doc {meta.get('project_name', 'unknown')}: "
                    f"{field}={actual} (expected {expected})"
                )
                is_valid = False
                break

        # Check numeric filters
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
        for w in warnings[:5]:  # Limit to first 5
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

    for field, value in filters.equality_filters.items():
        parts.append(f"{field}={value}")

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

"""
Query Filter Extractor - Extract ChromaDB filters from natural language queries
================================================================================

Handles:
1. Numeric comparisons: "> $100/kW", "above 50", "under $50/kW"
2. Zone filters: "West Texas", "coastal", "panhandle area"
3. Fuel type filters: "battery storage", "solar projects", "wind farms"
4. TSP filters: "ONCOR", "Centerpoint", "ETT"
5. Developer filters: "RWE projects", "Samsung battery"

Returns ChromaDB-compatible where clauses for hard pre-filtering.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


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


# Zone keyword mappings
ZONE_KEYWORDS = {
    "WEST": ["west texas", "west tx", "western texas", "far west", "permian"],
    "COAST": ["coast", "coastal", "houston", "gulf", "galveston", "brazoria"],
    "NORTH": ["north texas", "north tx", "northern texas", "dallas", "dfw", "fort worth"],
    "SOUTH": ["south texas", "south tx", "southern texas", "rio grande", "laredo"],
    "PANHANDLE": ["panhandle", "amarillo", "lubbock"],
    "CENTRAL": ["central texas", "central tx", "austin", "san antonio", "hill country"],
}

# Fuel type keyword mappings
FUEL_TYPE_KEYWORDS = {
    "OTH": ["battery", "bess", "storage", "energy storage", "battery storage"],
    "SOL": ["solar", "pv", "photovoltaic", "solar farm", "solar project"],
    "WIN": ["wind", "wind farm", "wind project", "wind turbine"],
    "GAS": ["gas", "natural gas", "ccgt", "combustion turbine", "peaker"],
}

# TSP keyword mappings (normalized names)
TSP_KEYWORDS = {
    "ONCOR": ["oncor"],
    "CENTERPOINT": ["centerpoint", "center point", "cnp"],
    "AEP TEXAS": ["aep", "aep texas"],
    "TNMP": ["tnmp", "texas-new mexico"],
    "ETT": ["ett", "electric transmission texas"],
    "LCRA TSC": ["lcra", "lower colorado"],
}

# Numeric field patterns
# Matches: "> $100/kW", "above 100", "more than $50", "greater than 50/kw"
# Matches: "< $50/kW", "under 100", "less than $50", "below 50/kw"
NUMERIC_PATTERNS = [
    # Pattern: "> $100/kW" or ">$100/kw" or "> 100/kW"
    (r'(?:>|greater\s+than|above|more\s+than|over|exceeds?)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:/kw|per\s*kw)?',
     '$gt', 'security_per_kw'),
    (r'(?:>=|at\s+least)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:/kw|per\s*kw)?',
     '$gte', 'security_per_kw'),
    # Pattern: "< $50/kW" or "<$50/kw" or "< 50/kW"
    (r'(?:<|less\s+than|under|below)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:/kw|per\s*kw)?',
     '$lt', 'security_per_kw'),
    (r'(?:<=|at\s+most|up\s+to)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:/kw|per\s*kw)?',
     '$lte', 'security_per_kw'),
    # Capacity patterns: "above 100 MW", "> 200 mw"
    (r'(?:>|greater\s+than|above|more\s+than|over)\s*(\d+(?:\.\d+)?)\s*(?:mw|megawatt)',
     '$gt', 'capacity_mw'),
    (r'(?:<|less\s+than|under|below)\s*(\d+(?:\.\d+)?)\s*(?:mw|megawatt)',
     '$lt', 'capacity_mw'),
]


def extract_zone(query: str) -> Optional[str]:
    """Extract zone from query using keyword matching."""
    q_lower = query.lower()

    for zone, keywords in ZONE_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                return zone

    return None


def extract_fuel_type(query: str) -> Optional[str]:
    """Extract fuel type from query using keyword matching."""
    q_lower = query.lower()

    for fuel_type, keywords in FUEL_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                return fuel_type

    return None


def extract_tsp(query: str) -> Optional[str]:
    """Extract TSP from query using keyword matching."""
    q_lower = query.lower()

    for tsp, keywords in TSP_KEYWORDS.items():
        for kw in keywords:
            # Use word boundary to avoid partial matches
            if re.search(rf'\b{re.escape(kw)}\b', q_lower):
                return tsp

    return None


def extract_developer(query: str) -> Optional[str]:
    """Extract developer/parent company from query."""
    # Import here to avoid circular imports
    from src.chunks.metadata import PARENT_MAPPING

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
    """
    equality_filters = {}
    numeric_filters = []

    # Extract zone
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

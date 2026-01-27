"""
Alias Expander - Expand developer/TSP short names to all ChromaDB-stored variants.

This bridges the gap between user queries (e.g., "RWE") and corpus values
(e.g., "RWE SOLAR DEVELOPMENT, LLC").

The issue: ChromaDB uses exact string matching for metadata filters.
When user asks about "RWE", we need to expand to all variants stored in the corpus.
"""

from typing import List, Dict, Set


# =============================================================================
# PARENT COMPANY ALIASES
# Map canonical names to ALL variants that might be stored in ChromaDB
# These must match EXACTLY what's in your corpus metadata
# =============================================================================

CHROMADB_PARENT_ALIASES: Dict[str, List[str]] = {
    # Tier 1: The Giants
    'RWE': [
        'RWE',
        'RWE SOLAR DEVELOPMENT',
        'RWE SOLAR DEVELOPMENT, LLC',
        'RWE RENEWABLES',
        'RWE RENEWABLES AMERICAS',
        'RWE RENEWABLES AMERICAS, LLC',
        'RWE RENEWABLES DEVELOPMENT',
        'RWE RENEWABLES DEVELOPMENT, LLC',
        'RWE CLEAN ENERGY',
        'RWE CLEAN ENERGY, LLC',
        'E.ON',
        'INLAND',
    ],
    'SAMSUNG': [
        'SAMSUNG',
        'SAMSUNG C&T',
        'SAMSUNG C&T AMERICA',
        'SAMSUNG C&T AMERICA, INC.',
        'SAMSUNG C&T AMERICA, INC',
        'SAMSUNG C&T CORPORATION',
        'SAMSUNG SDI',
        'SAMSUNG SDI AMERICA',
        'SAMSUNG SDI AMERICA, INC.',
    ],
    'NEXTERA': [
        'NEXTERA',
        'NEXTERA ENERGY',
        'NEXTERA ENERGY RESOURCES',
        'NEXTERA ENERGY RESOURCES, LLC',
        'NEXTERA ENERGY INTERCONNECTION HOLDINGS',
        'NEXTERA ENERGY INTERCONNECTION HOLDINGS, LLC',
        'FPL',
        'FPL GROUP',
        'FLORIDA POWER',
        'LOGAN P',
    ],
    'INVENERGY': [
        'INVENERGY',
        'INVENERGY LLC',
        'INVENERGY SERVICES',
        'INVENERGY SERVICES LLC',
        'INVENERGY SOLAR',
        'INVENERGY SOLAR LLC',
        'INVENERGY WIND',
        'INVENERGY WIND LLC',
    ],
    'EDF': [
        'EDF',
        'EDF RENEWABLES',
        'EDF RENEWABLES NORTH AMERICA',
        'EDF RENEWABLE ENERGY',
        'EDF RENEWABLE ENERGY, INC.',
        'EDF RENEWABLES, INC.',
    ],
    'ENEL': [
        'ENEL',
        'ENEL GREEN POWER',
        'ENEL GREEN POWER NORTH AMERICA',
        'ENEL GREEN POWER NORTH AMERICA, INC.',
        'ENEL X',
    ],
    'AES': [
        'AES',
        'AES CORPORATION',
        'AES CLEAN ENERGY',
        'AES CLEAN ENERGY, LLC',
        'AES SOLAR',
    ],
    'ENGIE': [
        'ENGIE',
        'ENGIE NORTH AMERICA',
        'ENGIE NORTH AMERICA INC.',
        'ENGIE SOLAR',
    ],
    'ORSTED': [
        'ORSTED',
        'ØRSTED',
        'ORSTED NORTH AMERICA',
        'ORSTED ONSHORE',
    ],
    'AVANGRID': [
        'AVANGRID',
        'AVANGRID RENEWABLES',
        'AVANGRID RENEWABLES, LLC',
    ],
    'PATTERN': [
        'PATTERN',
        'PATTERN ENERGY',
        'PATTERN ENERGY GROUP',
    ],

    # Tier 2: Major Players
    'CANADIAN SOLAR': [
        'CANADIAN SOLAR',
        'RECURRENT',
        'RECURRENT ENERGY',
        'RECURRENT ENERGY, LLC',
    ],
    'LIGHTSOURCE BP': [
        'LIGHTSOURCE',
        'LIGHTSOURCE BP',
        'BP',
        'BP SOLAR',
    ],
    'INTERSECT': [
        'INTERSECT',
        'INTERSECT POWER',
        'INTERSECT POWER, LLC',
    ],
    'SAVION': [
        'SAVION',
        'SAVION LLC',
    ],
    'CLEARWAY': [
        'CLEARWAY',
        'CLEARWAY ENERGY',
        'CLEARWAY ENERGY GROUP',
    ],
    'APEX': [
        'APEX',
        'APEX CLEAN ENERGY',
        'APEX CLEAN ENERGY, INC.',
    ],
    'HECATE': [
        'HECATE',
        'HECATE ENERGY',
        'HECATE ENERGY LLC',
        'HECATE ENERGY, LLC',
    ],
    'LEEWARD': [
        'LEEWARD',
        'LEEWARD RENEWABLE ENERGY',
    ],
    'LONGROAD': [
        'LONGROAD',
        'LONGROAD ENERGY',
    ],
    'CYPRESS CREEK': [
        'CYPRESS CREEK',
        'CYPRESS CREEK RENEWABLES',
    ],
    '174 POWER': [
        '174 POWER',
        '174 POWER GLOBAL',
    ],
    'ADAPTURE': [
        'ADAPTURE',
        'ADAPTURE RENEWABLES',
    ],

    # Tier 3: Battery/Storage Specialists
    'PLUS POWER': [
        'PLUS POWER',
        'PLUS POWER LLC',
        'PLUS POWER, LLC',
    ],
    'KEY CAPTURE': [
        'KEY CAPTURE',
        'KEY CAPTURE ENERGY',
    ],
    'BROAD REACH': [
        'BROAD REACH',
        'BROAD REACH POWER',
    ],
    'JUPITER': [
        'JUPITER',
        'JUPITER POWER',
        'JUPITER POWER LLC',
    ],
    'ABLE GRID': [
        'ABLE GRID',
        'ABLE GRID ENERGY',
    ],

    # Tier 4: Utilities & IPPs
    'VISTRA': [
        'VISTRA',
        'VISTRA CORP',
        'VISTRA ENERGY',
    ],
    'NRG': [
        'NRG',
        'NRG ENERGY',
        'NRG ENERGY, INC.',
    ],
    'DUKE': [
        'DUKE',
        'DUKE ENERGY',
        'DUKE ENERGY RENEWABLES',
    ],
    'SOUTHERN': [
        'SOUTHERN',
        'SOUTHERN COMPANY',
        'SOUTHERN POWER',
        'SOUTHERN POWER COMPANY',
    ],

    # Tier 5: Regional/Other
    'ORIGIS': [
        'ORIGIS',
        'ORIGIS ENERGY',
    ],
    'SOL SYSTEMS': [
        'SOL SYSTEMS',
        'SOL SYSTEMS LLC',
    ],
    'TRI GLOBAL': [
        'TRI GLOBAL',
        'TRI-GLOBAL',
        'TRI GLOBAL ENERGY',
    ],
    '8MINUTE': [
        '8MINUTE',
        '8 MINUTE',
        '8MINUTE SOLAR ENERGY',
    ],
    'IP ENERGY': [
        'IP ENERGY',
        'IP QUANTUM',
    ],
}


# =============================================================================
# TSP ALIASES
# Map TSP short names to all known variants in ChromaDB
# =============================================================================

CHROMADB_TSP_ALIASES: Dict[str, List[str]] = {
    'ONCOR': [
        'ONCOR',
        'ONCOR ELECTRIC',
        'ONCOR ELECTRIC DELIVERY',
        'ONCOR ELECTRIC DELIVERY COMPANY',
        'ONCOR ELECTRIC DELIVERY COMPANY LLC',
        'ONCOR ELECTRIC DELIVERY COMPANY, LLC',
    ],
    'CENTERPOINT': [
        'CENTERPOINT',
        'CENTERPOINT ENERGY',
        'CENTERPOINT ENERGY HOUSTON',
        'CENTERPOINT ENERGY HOUSTON ELECTRIC',
        'CENTERPOINT ENERGY HOUSTON ELECTRIC, LLC',
        'CNP',
        'CPNT',
    ],
    'AEP': [
        'AEP',
        'AEP TEXAS',
        'AEP TEXAS INC',
        'AEP TEXAS INC.',
        'AEP TEXAS CENTRAL',
        'AEP TEXAS CENTRAL COMPANY',
        'AEP TEXAS NORTH',
        'AEP TEXAS NORTH COMPANY',
        'SWEPCO',
    ],
    'TNMP': [
        'TNMP',
        'TEXAS-NEW MEXICO POWER',
        'TEXAS-NEW MEXICO POWER COMPANY',
        'TEXAS NEW MEXICO POWER',
    ],
    'ETT': [
        'ETT',
        'ELECTRIC TRANSMISSION TEXAS',
        'ELECTRIC TRANSMISSION TEXAS, LLC',
        'ELECTRIC TRANSMISSION TEXAS LLC',
    ],
    'LCRA': [
        'LCRA',
        'LCRA TSC',
        'LCRA TRANSMISSION SERVICES',
        'LOWER COLORADO RIVER AUTHORITY',
    ],
    'SHARYLAND': [
        'SHARYLAND',
        'SHARYLAND UTILITIES',
        'SHARYLAND UTILITIES, L.P.',
    ],
}


# =============================================================================
# EXPANSION FUNCTIONS
# =============================================================================

def expand_parent_company_aliases(companies: List[str]) -> List[str]:
    """
    Expand short developer names to all known ChromaDB variants.

    This is critical for comparative queries like "RWE vs SAMSUNG" where
    the user uses short names but ChromaDB stores full legal names.

    Args:
        companies: List of company names (e.g., ['RWE', 'SAMSUNG'])

    Returns:
        Expanded list of all variants (e.g., ['RWE', 'RWE SOLAR DEVELOPMENT',
        'RWE SOLAR DEVELOPMENT, LLC', 'SAMSUNG', 'SAMSUNG C&T AMERICA, INC.', ...])

    Example:
        >>> expand_parent_company_aliases(['RWE', 'SAMSUNG'])
        ['RWE', 'RWE SOLAR DEVELOPMENT', 'RWE SOLAR DEVELOPMENT, LLC',
         'RWE RENEWABLES', ..., 'SAMSUNG', 'SAMSUNG C&T', ...]
    """
    if not companies:
        return []

    expanded = []
    for company in companies:
        company_upper = company.upper().strip()

        # Check if we have aliases for this company
        matched = False
        for canonical, aliases in CHROMADB_PARENT_ALIASES.items():
            # Match if input equals canonical name OR is in the alias list
            aliases_upper = [a.upper() for a in aliases]
            if company_upper == canonical or company_upper in aliases_upper:
                expanded.extend(aliases)
                matched = True
                break

        # If no alias found, include the original input
        if not matched:
            expanded.append(company)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    result = []
    for item in expanded:
        item_upper = item.upper()
        if item_upper not in seen:
            seen.add(item_upper)
            result.append(item)

    return result


def expand_tsp_aliases(tsps: List[str]) -> List[str]:
    """
    Expand short TSP names to all known ChromaDB variants.

    Args:
        tsps: List of TSP names (e.g., ['ONCOR', 'CENTERPOINT'])

    Returns:
        Expanded list of all variants
    """
    if not tsps:
        return []

    expanded = []
    for tsp in tsps:
        tsp_upper = tsp.upper().strip()

        matched = False
        for canonical, aliases in CHROMADB_TSP_ALIASES.items():
            aliases_upper = [a.upper() for a in aliases]
            if tsp_upper == canonical or tsp_upper in aliases_upper:
                expanded.extend(aliases)
                matched = True
                break

        if not matched:
            expanded.append(tsp)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    result = []
    for item in expanded:
        item_upper = item.upper()
        if item_upper not in seen:
            seen.add(item_upper)
            result.append(item)

    return result


def get_canonical_name(company: str, mapping: Dict[str, List[str]]) -> str:
    """
    Get the canonical (short) name for a company variant.

    Args:
        company: A company name variant (e.g., "RWE SOLAR DEVELOPMENT, LLC")
        mapping: The alias mapping to use

    Returns:
        Canonical name (e.g., "RWE") or original if not found
    """
    company_upper = company.upper().strip()

    for canonical, aliases in mapping.items():
        aliases_upper = [a.upper() for a in aliases]
        if company_upper == canonical or company_upper in aliases_upper:
            return canonical

    return company


def get_canonical_parent(company: str) -> str:
    """Get canonical parent company name."""
    return get_canonical_name(company, CHROMADB_PARENT_ALIASES)


def get_canonical_tsp(tsp: str) -> str:
    """Get canonical TSP name."""
    return get_canonical_name(tsp, CHROMADB_TSP_ALIASES)


# =============================================================================
# RUNTIME ALIAS UPDATER
# For cases where corpus inspection reveals new variants
# =============================================================================

def add_parent_alias(canonical: str, new_alias: str) -> None:
    """
    Add a new alias for a parent company at runtime.

    This is useful after running corpus verification to add variants
    that weren't initially anticipated.

    Args:
        canonical: The canonical company name (e.g., 'RWE')
        new_alias: The new alias to add (e.g., 'RWE CLEAN POWER, LLC')
    """
    if canonical in CHROMADB_PARENT_ALIASES:
        if new_alias not in CHROMADB_PARENT_ALIASES[canonical]:
            CHROMADB_PARENT_ALIASES[canonical].append(new_alias)
    else:
        CHROMADB_PARENT_ALIASES[canonical] = [canonical, new_alias]


def add_tsp_alias(canonical: str, new_alias: str) -> None:
    """Add a new alias for a TSP at runtime."""
    if canonical in CHROMADB_TSP_ALIASES:
        if new_alias not in CHROMADB_TSP_ALIASES[canonical]:
            CHROMADB_TSP_ALIASES[canonical].append(new_alias)
    else:
        CHROMADB_TSP_ALIASES[canonical] = [canonical, new_alias]

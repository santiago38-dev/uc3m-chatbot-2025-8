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
# Updated based on verify_corpus_developers.py output
# =============================================================================

CHROMADB_PARENT_ALIASES: Dict[str, List[str]] = {
    # === VERIFIED FROM CORPUS (exact values) ===
    'RWE': [
        'RWE',
        'RWE Clean Energy Development, LLC',  # 69 chunks in corpus
        'RWE Solar Development, LLC',          # 68 chunks in corpus
    ],
    'SAMSUNG': [
        'SAMSUNG',                             # 63 chunks in corpus
        'SAMSUNG C&T',                         # 77 chunks in corpus
    ],
    'INTERSECT': [
        'INTERSECT',                           # 150 chunks in corpus
    ],
    'ENGIE': [
        'ENGIE',                               # 207 chunks in corpus
    ],
    'SUNCHASE': [
        'SUNCHASE',                            # 207 chunks in corpus
    ],
    'CENTERPOINT ENERGY': [
        'CENTERPOINT ENERGY',                  # 153 chunks in corpus
    ],
    'NRG': [
        'NRG',
        'NRG THW GT LLC',                      # 159 chunks in corpus
    ],

    # === ADDITIONAL COMMON ALIASES (may need verification) ===
    'NEXTERA': [
        'NEXTERA',
        'NEXTERA ENERGY',
        'NEXTERA ENERGY RESOURCES',
        'NEXTERA ENERGY RESOURCES, LLC',
        'FPL',
        'FPL GROUP',
    ],
    'INVENERGY': [
        'INVENERGY',
        'INVENERGY LLC',
        'INVENERGY SERVICES',
    ],
    'EDF': [
        'EDF',
        'EDF RENEWABLES',
        'EDF RENEWABLES NORTH AMERICA',
    ],
    'ENEL': [
        'ENEL',
        'ENEL GREEN POWER',
        'ENEL GREEN POWER NORTH AMERICA',
    ],
    'AES': [
        'AES',
        'AES CORPORATION',
        'AES CLEAN ENERGY',
    ],
    'ORSTED': [
        'ORSTED',
        'ØRSTED',
        'ORSTED NORTH AMERICA',
    ],
    'AVANGRID': [
        'AVANGRID',
        'AVANGRID RENEWABLES',
    ],
    'PATTERN': [
        'PATTERN',
        'PATTERN ENERGY',
    ],
    'CANADIAN SOLAR': [
        'CANADIAN SOLAR',
        'RECURRENT',
        'RECURRENT ENERGY',
    ],
    'LIGHTSOURCE BP': [
        'LIGHTSOURCE',
        'LIGHTSOURCE BP',
    ],
    'SAVION': [
        'SAVION',
        'SAVION LLC',
    ],
    'CLEARWAY': [
        'CLEARWAY',
        'CLEARWAY ENERGY',
    ],
    'APEX': [
        'APEX',
        'APEX CLEAN ENERGY',
    ],
    'HECATE': [
        'HECATE',
        'HECATE ENERGY',
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
    'PLUS POWER': [
        'PLUS POWER',
        'PLUS POWER LLC',
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
    ],
    'VISTRA': [
        'VISTRA',
        'VISTRA CORP',
        'VISTRA ENERGY',
    ],
    'DUKE': [
        'DUKE',
        'DUKE ENERGY',
    ],
    'SOUTHERN': [
        'SOUTHERN',
        'SOUTHERN COMPANY',
        'SOUTHERN POWER',
    ],
    'ORIGIS': [
        'ORIGIS',
        'ORIGIS ENERGY',
    ],
    '8MINUTE': [
        '8MINUTE',
        '8 MINUTE',
        '8MINUTE SOLAR ENERGY',
    ],
}


# =============================================================================
# TSP ALIASES
# Map TSP short names to all known variants in ChromaDB
# Updated based on verify_corpus_developers.py output
# =============================================================================

CHROMADB_TSP_ALIASES: Dict[str, List[str]] = {
    # === VERIFIED FROM CORPUS (exact values) ===
    'ONCOR': [
        'ONCOR',                               # 2579 chunks in corpus
    ],
    'CENTERPOINT': [
        'CENTERPOINT',                         # 4779 chunks in corpus
    ],
    'AEP': [
        'AEP',                                  # 1240 chunks in corpus
    ],
    'TNMP': [
        'TNMP',                                 # 1494 chunks in corpus
    ],
    'ETT': [
        'ETT',                                  # 555 chunks in corpus
    ],
    'LCRA': [
        'LCRA',                                 # 444 chunks in corpus
    ],
    'LONE STAR': [
        'LONE STAR',                           # 171 chunks in corpus
    ],
    'CPS': [
        'CPS',                                  # 152 chunks in corpus
    ],
    'BRAZOS': [
        'BRAZOS',                              # 66 chunks in corpus
    ],
    'RAYBURN': [
        'Rayburn Country Electric Cooperative, Inc.',  # 121 chunks in corpus
    ],

    # === ADDITIONAL ALIASES (for query flexibility) ===
    'SHARYLAND': [
        'SHARYLAND',
        'SHARYLAND UTILITIES',
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

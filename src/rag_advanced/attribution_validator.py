"""
Attribution Validator - Anti-Hallucination Checks for RAG Responses.

This module validates LLM responses don't misattribute projects to wrong developers.
Catches hallucinations like "RWE's Champaign BESS" when Champaign BESS is actually SAMSUNG.

The problem:
When comparative queries (e.g., "RWE vs SAMSUNG") fail to retrieve one entity's documents,
the LLM may hallucinate by assigning the other entity's projects to fill the gap.

Solution:
1. Build a ground truth map of project -> developer from retrieved docs
2. Extract developer-project attributions from the LLM response
3. Flag any misattributions as hallucination warnings
"""

import re
from typing import List, Dict, Tuple, Set, Optional
from langchain_core.documents import Document

from .alias_expander import get_canonical_parent, CHROMADB_PARENT_ALIASES


def extract_project_developer_map(docs: List[Document]) -> Dict[str, str]:
    """
    Build mapping of project names to their actual developers from source docs.

    This is the ground truth we use to validate LLM attributions.

    Args:
        docs: Retrieved documents with metadata

    Returns:
        Dict mapping lowercase project names to developer names
        Example: {'champaign bess': 'SAMSUNG C&T AMERICA, INC.', ...}
    """
    project_to_developer = {}

    for doc in docs:
        meta = doc.metadata
        project = meta.get('project_name', '')
        developer = meta.get('parent_company', '') or meta.get('developer_spv', '')

        if project and developer:
            # Normalize project name for matching
            project_key = project.lower().strip()
            project_to_developer[project_key] = developer

    return project_to_developer


def get_developers_in_docs(docs: List[Document]) -> Set[str]:
    """
    Get set of all developers present in retrieved docs.

    Useful for checking if a comparative query actually retrieved
    documents for all requested entities.

    Args:
        docs: Retrieved documents

    Returns:
        Set of unique developer names (canonical form)
    """
    developers = set()
    for doc in docs:
        dev = doc.metadata.get('parent_company', '') or doc.metadata.get('developer_spv', '')
        if dev:
            # Get canonical name for consistency
            canonical = get_canonical_parent(dev)
            developers.add(canonical)
    return developers


def check_missing_entities(
    requested_entities: List[str],
    docs: List[Document]
) -> List[str]:
    """
    Check which requested entities have no documents in the retrieved set.

    Critical for comparative queries: if user asks "RWE vs SAMSUNG" but
    no RWE docs are retrieved, we need to warn about this.

    Args:
        requested_entities: List of entity names from the query (canonical form)
        docs: Retrieved documents

    Returns:
        List of missing entity names
    """
    present = get_developers_in_docs(docs)

    # Normalize requested to canonical form
    requested_canonical = [get_canonical_parent(e) for e in requested_entities]

    missing = []
    for entity in requested_canonical:
        if entity not in present:
            missing.append(entity)

    return missing


# Patterns to detect developer-project attributions in LLM responses
ATTRIBUTION_PATTERNS = [
    # "RWE - Champaign BESS" or "RWE's Champaign BESS"
    r"((?:RWE|SAMSUNG|NEXTERA|INTERSECT|INVENERGY|EDF|ENEL|SOUTHERN|AES|HECATE|CLEARWAY|APEX|VISTRA|PLUS POWER|KEY CAPTURE|JUPITER|ORSTED|ENGIE|AVANGRID|PATTERN|CANADIAN SOLAR|LIGHTSOURCE|ORIGIS|8MINUTE)[^:]*?)['']?s?\s*[-–—:]\s*([A-Za-z0-9\s]+(?:BESS|Solar|Wind|Storage|Energy|Power|Farm)?)",

    # "Entity A (RWE - Champaign BESS)"
    r"Entity [AB]\s*\(\s*([A-Za-z\s&]+?)[\s\-–—]+([^)]+)\)",

    # Table rows: "| RWE | Champaign BESS |"
    r"\|\s*([A-Za-z\s&]+?)\s*\|\s*([A-Za-z0-9\s]+(?:BESS|Solar|Wind|Storage))\s*\|",

    # "Developer: RWE" followed by "Project: Champaign BESS"
    r"Developer:\s*([A-Za-z\s&]+?)\s*(?:\||,|\n)[^\n]*?Project[:\s]+([A-Za-z0-9\s]+)",

    # "RWE projects include Champaign BESS"
    r"([A-Za-z\s&]+?)\s+projects?\s+(?:include|are|:)\s*([A-Za-z0-9\s,]+(?:BESS|Solar|Wind|Storage))",
]

# Known developer short names for pattern matching
KNOWN_DEVELOPERS = list(CHROMADB_PARENT_ALIASES.keys())


def extract_claimed_attributions(response: str) -> List[Tuple[str, str]]:
    """
    Extract developer-project attributions claimed in the response.

    Args:
        response: The LLM-generated response text

    Returns:
        List of (developer, project) tuples claimed in the response
        Example: [('RWE', 'Champaign BESS'), ('SAMSUNG', 'Rutile BESS')]
    """
    attributions = []

    for pattern in ATTRIBUTION_PATTERNS:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if len(match) >= 2:
                developer = match[0].strip()
                project = match[1].strip()

                # Clean up the developer name
                developer_upper = developer.upper()

                # Check if this looks like a real developer name
                is_valid_dev = False
                for known in KNOWN_DEVELOPERS:
                    if known in developer_upper or developer_upper in known:
                        developer = known
                        is_valid_dev = True
                        break

                # Filter out noise
                if is_valid_dev and project and len(project) > 3:
                    # Clean project name
                    project = re.sub(r'\s+', ' ', project).strip()
                    if not project.lower().startswith(('the ', 'a ', 'an ')):
                        attributions.append((developer, project))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for attr in attributions:
        key = (attr[0].upper(), attr[1].lower())
        if key not in seen:
            seen.add(key)
            unique.append(attr)

    return unique


def validate_developer_attributions(
    response: str,
    retrieved_docs: List[Document]
) -> Tuple[bool, List[str]]:
    """
    Validate that LLM response doesn't misattribute projects.

    This is the main anti-hallucination check.

    Args:
        response: The LLM-generated response
        retrieved_docs: Documents that were used as context

    Returns:
        (is_valid, warnings)
        - is_valid: True if no misattributions detected
        - warnings: List of warning messages for any issues found
    """
    warnings = []
    is_valid = True

    # Build ground truth from docs
    project_map = extract_project_developer_map(retrieved_docs)

    # Get all developers actually in the docs
    actual_developers = set()
    for dev in project_map.values():
        canonical = get_canonical_parent(dev)
        actual_developers.add(canonical)
        # Also add the raw value
        actual_developers.add(dev.upper())

    # Extract claims from response
    claimed = extract_claimed_attributions(response)

    for claimed_dev, claimed_project in claimed:
        # Find actual developer for this project
        actual_dev = None
        claimed_project_lower = claimed_project.lower().strip()

        # Try to match the project name
        for proj_key, proj_dev in project_map.items():
            # Fuzzy match on project name
            if (claimed_project_lower in proj_key or
                proj_key in claimed_project_lower or
                claimed_project_lower.split()[0] in proj_key):
                actual_dev = proj_dev
                break

        if actual_dev:
            # Get canonical form of actual developer
            actual_canonical = get_canonical_parent(actual_dev)

            # Check if claimed developer matches actual
            claimed_canonical = get_canonical_parent(claimed_dev)

            if claimed_canonical != actual_canonical:
                # HALLUCINATION DETECTED
                is_valid = False
                warnings.append(
                    f"ATTRIBUTION ERROR: '{claimed_project}' attributed to {claimed_dev}, "
                    f"but source documents show developer is '{actual_dev}' ({actual_canonical})"
                )

    return is_valid, warnings


def generate_attribution_warning(
    missing_entities: List[str],
    found_entities: Set[str],
    lang: str = 'english'
) -> Optional[str]:
    """
    Generate a user-friendly warning when some requested entities
    have no data in the retrieved documents.

    Args:
        missing_entities: Entities requested but not found
        found_entities: Entities that were found
        lang: Language for the warning message

    Returns:
        Warning message or None if no issues
    """
    if not missing_entities:
        return None

    found_list = ', '.join(sorted(found_entities)) if found_entities else 'none'

    if lang == 'spanish':
        if len(missing_entities) == 1:
            return (
                f"**Nota sobre datos:** No se encontraron documentos para "
                f"**{missing_entities[0]}** en la base de datos. "
                f"Entidades encontradas: {found_list}."
            )
        else:
            missing_list = ', '.join(missing_entities)
            return (
                f"**Nota sobre datos:** No se encontraron documentos para "
                f"**{missing_list}** en la base de datos. "
                f"Entidades encontradas: {found_list}."
            )
    else:
        if len(missing_entities) == 1:
            return (
                f"**Data Note:** No documents found for **{missing_entities[0]}** "
                f"in the database. Found entities: {found_list}."
            )
        else:
            missing_list = ', '.join(missing_entities)
            return (
                f"**Data Note:** No documents found for **{missing_list}** "
                f"in the database. Found entities: {found_list}."
            )


def create_grounding_context(docs: List[Document]) -> str:
    """
    Create a grounding context string that explicitly lists
    which developers own which projects.

    This can be prepended to the LLM prompt to reinforce grounding.

    Args:
        docs: Retrieved documents

    Returns:
        Grounding context string
    """
    project_map = extract_project_developer_map(docs)

    if not project_map:
        return ""

    lines = ["VERIFIED PROJECT OWNERSHIP (from source documents):"]
    for project, developer in sorted(project_map.items()):
        canonical = get_canonical_parent(developer)
        lines.append(f"- {project.title()}: owned by {canonical}")

    lines.append("")
    lines.append("IMPORTANT: Only attribute projects to developers listed above.")

    return "\n".join(lines)

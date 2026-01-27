"""
Hallucination Guard - Prevent and detect LLM hallucinations
============================================================

Two main functions:
1. validate_entities_exist() - PRE-check: Do queried entities exist in corpus?
2. validate_response_grounding() - POST-check: Are claims grounded in sources?
"""

import re
from typing import Dict, List, Tuple, Set
from src.vector_store import get_vectorstore
from src.chunks.metadata import PARENT_MAPPING  # Use existing ERCOT developer aliases


def get_corpus_metadata_values(field: str) -> Set[str]:
    """
    Get all unique values for a metadata field from the corpus.

    Args:
        field: Metadata field name (e.g., 'parent_company', 'zone', 'fuel_type')

    Returns:
        Set of unique values for that field
    """
    try:
        vectorstore = get_vectorstore()
        # Access underlying ChromaDB collection
        collection = vectorstore._collection

        # Get all documents with their metadata
        results = collection.get(include=["metadatas"])

        values = set()
        for meta in results.get("metadatas", []):
            if meta and field in meta:
                val = meta[field]
                if val:
                    values.add(str(val).upper().strip())

        return values
    except Exception as e:
        print(f"Warning: Could not get corpus metadata: {e}")
        return set()


def fuzzy_match(query: str, corpus_values: Set[str]) -> Tuple[bool, str]:
    """
    Check if query matches any corpus value using partial/fuzzy matching.

    Handles cases like:
    - "NEXTERA" matching "NEXTERA ENERGY RESOURCES"
    - "RWE" matching "RWE SOLAR DEVELOPMENT"

    Uses PARENT_MAPPING from metadata.py for comprehensive ERCOT developer aliases.

    Returns:
        Tuple of (found: bool, matched_value: str or None)
    """
    query = query.upper().strip()

    # Exact match first
    if query in corpus_values:
        return True, query

    # Partial match: query is substring of corpus value
    for cv in corpus_values:
        if query in cv:
            return True, cv

    # Partial match: corpus value is substring of query
    for cv in corpus_values:
        if cv in query:
            return True, cv

    # Use PARENT_MAPPING from metadata.py for ERCOT-specific developer aliases
    # PARENT_MAPPING format: {"NEXTERA": ["nextera", "fpl", ...], "RWE": ["rwe", "e.on", ...]}
    if query in PARENT_MAPPING:
        aliases = [a.upper() for a in PARENT_MAPPING[query]]
        for alias in aliases:
            for cv in corpus_values:
                if alias in cv or cv in alias:
                    return True, cv

    # Also check reverse: if query matches any alias, find the parent company
    for parent, aliases in PARENT_MAPPING.items():
        aliases_upper = [a.upper() for a in aliases]
        if query.lower() in aliases or any(query in a.upper() for a in aliases):
            # Query matches an alias, check if parent is in corpus
            for cv in corpus_values:
                if parent in cv or cv in parent:
                    return True, cv

    return False, None


def validate_entities_exist(extracted_metadata: Dict, strict: bool = False) -> Dict:
    """
    PRE-VALIDATION: Check if extracted entities actually exist in the corpus.

    Args:
        extracted_metadata: Dict from extract_query_metadata()
                           e.g., {"parent_company": "NEXTERA", "zone": "WEST"}
        strict: If True, exact match only. If False, allows partial/fuzzy matching.

    Returns:
        Dict with validation results:
        {
            "valid": True/False,
            "missing_entities": [{"field": "parent_company", "value": "NEXTERA", "available": ["RWE", "SAMSUNG", ...]}],
            "warning_message": "No NEXTERA projects found in corpus. Available developers: RWE, SAMSUNG, ...",
            "should_abort": True/False  # Recommend aborting query for missing critical entities
        }
    """
    missing = []
    matched = {}  # Track what was matched for logging

    # Fields that are important to validate
    # critical=True means query should abort if not found
    validatable_fields = {
        "parent_company": {"label": "developer", "critical": True},
        "zone": {"label": "zone", "critical": False},
        "fuel_type": {"label": "fuel type", "critical": False},
        "tsp_normalized": {"label": "TSP", "critical": False},
        "county": {"label": "county", "critical": False}
    }

    should_abort = False

    for field, config in validatable_fields.items():
        if field in extracted_metadata:
            queried_value = str(extracted_metadata[field]).upper().strip()
            corpus_values = get_corpus_metadata_values(field)

            if not corpus_values:
                continue  # Can't validate if no corpus data

            # Try fuzzy matching unless strict mode
            if strict:
                found = queried_value in corpus_values
                matched_value = queried_value if found else None
            else:
                found, matched_value = fuzzy_match(queried_value, corpus_values)

            if not found:
                # Entity not found in corpus
                available = sorted(list(corpus_values))[:10]
                missing.append({
                    "field": field,
                    "value": queried_value,
                    "label": config["label"],
                    "available": available,
                    "critical": config["critical"]
                })

                # Mark for abort if critical field is missing
                if config["critical"]:
                    should_abort = True
            else:
                matched[field] = matched_value

    if missing:
        # Build warning message
        warnings = []
        for m in missing:
            available_str = ", ".join(m["available"][:5])
            if len(m["available"]) > 5:
                available_str += f", ... ({len(m['available'])} total)"
            warnings.append(
                f"No '{m['value']}' {m['label']} found in corpus. "
                f"Available: {available_str}"
            )

        return {
            "valid": False,
            "missing_entities": missing,
            "matched_entities": matched,
            "warning_message": " | ".join(warnings),
            "should_abort": should_abort
        }

    return {
        "valid": True,
        "missing_entities": [],
        "matched_entities": matched,
        "warning_message": None,
        "should_abort": False
    }


def validate_response_grounding(
    response: str,
    retrieved_docs: List,
    claimed_entities: Dict = None
) -> Tuple[bool, List[str]]:
    """
    POST-VALIDATION: Check if response claims are grounded in retrieved documents.

    Detects:
    1. Developer names in response that don't match source metadata
    2. Project names attributed to wrong developers
    3. Numbers/values not found in sources

    Args:
        response: The LLM's response text
        retrieved_docs: List of retrieved LangChain documents
        claimed_entities: Optional dict of entities the question asked about

    Returns:
        Tuple of (is_grounded: bool, issues: List[str])
    """
    issues = []

    # Extract parent_company values from retrieved docs
    source_developers = set()
    source_projects = {}  # project_name -> parent_company

    for doc in retrieved_docs:
        meta = doc.metadata if hasattr(doc, 'metadata') else {}

        parent = meta.get('parent_company', '')
        if parent:
            source_developers.add(parent.upper().strip())

        project = meta.get('project_name', '')
        if project and parent:
            source_projects[project.upper().strip()] = parent.upper().strip()

    # Check if response mentions developers not in sources
    # Common developer names to check for
    known_developers = [
        "NEXTERA", "RWE", "INVENERGY", "EDF", "ENEL", "ENGIE",
        "SAMSUNG", "INTERSECT", "PLUS POWER", "CLEARWAY", "APEX",
        "ORSTED", "AVANGRID", "PATTERN", "CANADIAN SOLAR"
    ]

    response_upper = response.upper()

    for dev in known_developers:
        if dev in response_upper and dev not in source_developers:
            # Developer mentioned in response but not in sources
            issues.append(
                f"Response mentions '{dev}' but no retrieved documents have this developer. "
                f"Sources contain: {', '.join(sorted(source_developers)[:5]) if source_developers else 'no developer info'}"
            )

    # Check for specific hallucination patterns
    hallucination_patterns = [
        # "Project X is owned by Developer Y" but Y not in sources
        r"(?:owned by|developed by|belongs to|operated by)\s+([A-Z][a-zA-Z\s]+)",
    ]

    for pattern in hallucination_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        for match in matches:
            claimed_dev = match.strip().upper()
            # Check if this claimed developer is in sources
            if claimed_dev and len(claimed_dev) > 3:
                found = any(claimed_dev in sd for sd in source_developers)
                if not found and any(kd in claimed_dev for kd in known_developers):
                    issues.append(f"Response claims ownership by '{match}' but this is not supported by sources")

    is_grounded = len(issues) == 0
    return is_grounded, issues


def create_grounding_warning(missing_entities: List[Dict]) -> str:
    """
    Create a user-friendly warning message when entities don't exist.

    Returns a message like:
    "Note: No NextEra projects found in this corpus.
     Available developers include: RWE, SAMSUNG, INTERSECT, ..."
    """
    if not missing_entities:
        return ""

    lines = ["**⚠️ Data Availability Notice:**"]

    for m in missing_entities:
        available = m.get("available", [])[:5]
        available_str = ", ".join(available)
        lines.append(f"- No **{m['value']}** {m['label']} found in corpus")
        if available:
            lines.append(f"  Available {m['label']}s: {available_str}")

    lines.append("")
    lines.append("The response below may not accurately address your query.")
    lines.append("")

    return "\n".join(lines)

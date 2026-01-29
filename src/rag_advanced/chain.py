# Chain builders for RAG pipeline
#
# Enhanced with:
# - Analytics routing for aggregation queries (corpus-wide statistics)
# - Hard filtering for comparative queries (multi-value $in with alias expansion)
# - Deduplication by INR to prevent listing same projects multiple times
# - Attribution validation to catch hallucinations
# - filter_hook extension point for custom retrieval logic

import json
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union, Callable

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableGenerator
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.chat_history import get_session_history
from src.llm_client import call_llm_api_full
from .filter_utils import (
    extract_multi_filters_from_query,
    build_chromadb_where_clause
)
from .utils import (
    get_logger, detect_language, format_sources, RAGMode, config
)
from .prompts import REPHRASE_PROMPT
from .components import (
    is_domain_relevant, contextualize_question,
    generate_flash_response, generate_thinking_response,
    classify_query, deduplicate_docs_by_inr, get_unique_projects_from_docs
)
from .attribution_validator import (
    validate_developer_attributions,
    check_missing_entities,
    generate_attribution_warning,
    get_entities_in_docs,
    create_grounding_context
)
from .alias_expander import get_canonical_parent

# Default analytics path
DEFAULT_ANALYTICS_PATH = "data/corpus_analytics.json"

# Analytics-aware system prompt
ANALYTICS_SYSTEM_PROMPT_EN = """You are an ERCOT interconnection intelligence analyst. You have access to:

1. PRE-COMPUTED CORPUS ANALYTICS: Statistics calculated across ALL projects in the corpus
   - Use for: medians, rankings, distributions, market comparisons
   - These are AUTHORITATIVE - do not estimate or hallucinate different numbers

2. RETRIEVED DOCUMENT EXCERPTS: Specific text from interconnection agreements
   - Use for: legal language, specific project details, clause comparisons

When answering:
- For statistical questions, cite the analytics data explicitly
- For legal/contractual questions, quote from retrieved documents
- For comparative questions (e.g., "RWE vs market"), use BOTH sources
- Always clarify the data source: "Based on corpus analytics..." or "From the SGIA document..."

If the analytics don't contain data needed to answer, say so clearly rather than estimating.

IMPORTANT: All statistics include sample sizes (n=). When n < 10, note the limited sample size.
"""

ANALYTICS_SYSTEM_PROMPT_ES = """Eres un analista de inteligencia de interconexión de ERCOT. Tienes acceso a:

1. ANALÍTICAS DE CORPUS PRE-CALCULADAS: Estadísticas calculadas de TODOS los proyectos del corpus
   - Usar para: medianas, rankings, distribuciones, comparaciones de mercado
   - Estos son AUTORITATIVOS - no estimes ni inventes números diferentes

2. EXTRACTOS DE DOCUMENTOS RECUPERADOS: Texto específico de acuerdos de interconexión
   - Usar para: lenguaje legal, detalles específicos de proyectos, comparaciones de cláusulas

Al responder:
- Para preguntas estadísticas, cita los datos analíticos explícitamente
- Para preguntas legales/contractuales, cita los documentos recuperados
- Para preguntas comparativas (ej: "RWE vs mercado"), usa AMBAS fuentes
- Siempre aclara la fuente: "Según las analíticas del corpus..." o "Del documento SGIA..."

Si las analíticas no contienen los datos necesarios, dilo claramente en lugar de estimar.

IMPORTANTE: Todas las estadísticas incluyen tamaños de muestra (n=). Cuando n < 10, nota el tamaño de muestra limitado.
"""

# Out-of-scope question messages
OOS_QUESTION_MSG = {
    "english": ("This question is not related to ERCOT interconnection agreements. "
                "I can only answer questions about energy projects, power grids, and ERCOT."),
    "spanish": ("Esta pregunta no está relacionada con acuerdos de interconexión ERCOT. "
                "Solo puedo responder preguntas sobre proyectos de energía, redes eléctricas y ERCOT.")
}


# =============================================================================
# ANALYTICS FUNCTIONS
# =============================================================================

def load_analytics(path: str = DEFAULT_ANALYTICS_PATH) -> Optional[Dict]:
    """Load pre-computed analytics from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        get_logger().warning(f"Analytics file not found: {path}")
        return None
    except json.JSONDecodeError as e:
        get_logger().warning(f"Error parsing analytics: {e}")
        return None


def format_fuel_stats(fuel_data: Dict) -> str:
    """Format fuel type statistics for context."""
    lines = []
    for fuel_type, data in sorted(fuel_data.items(), key=lambda x: x[1].get("count", 0), reverse=True):
        if data.get("median_security_per_kw") is not None:
            lines.append(
                f"- {fuel_type}: {data['count']} projects, "
                f"median ${data['median_security_per_kw']}/kW (n={data.get('n', 'N/A')})"
            )
        else:
            lines.append(f"- {fuel_type}: {data['count']} projects (no security data)")
    return "\n".join(lines)


def format_zone_stats(zone_data: Dict) -> str:
    """Format zone statistics for context."""
    lines = []
    for zone, data in sorted(zone_data.items(), key=lambda x: x[1].get("count", 0), reverse=True):
        if data.get("median_security_per_kw") is not None:
            lines.append(
                f"- {zone}: {data['count']} projects, "
                f"median ${data['median_security_per_kw']}/kW (n={data.get('n', 'N/A')})"
            )
        else:
            lines.append(f"- {zone}: {data['count']} projects (no security data)")
    return "\n".join(lines)


def format_tsp_rankings(rankings: List[Dict], limit: int = 10) -> str:
    """Format TSP rankings for context."""
    lines = []
    for item in rankings[:limit]:
        lines.append(
            f"{item['rank']}. {item['tsp']}: avg ${item['avg_security_per_kw']}/kW "
            f"({item['project_count']} projects, n={item.get('n', 'N/A')})"
        )
    return "\n".join(lines)


def format_county_stats(county_data: List[Dict], limit: int = 10) -> str:
    """Format geographic concentration for context."""
    lines = []
    for item in county_data[:limit]:
        lines.append(
            f"- {item['county']} ({item['zone']}): {item['project_count']} projects"
        )
    return "\n".join(lines)


def format_developer_stats(dev_analysis: Dict) -> str:
    """Format developer analysis for context."""
    lines = []

    # Multi-zone developers - handles both formats:
    # New format: {dev: [zone_list]} or Old format: {dev: {project_count, zones}}
    multi_zone = dev_analysis.get("multi_zone_developers", {})
    if multi_zone:
        lines.append("**Multi-Zone Developers:**")
        for dev, data in list(multi_zone.items())[:5]:
            if isinstance(data, list):
                # New format: data is list of zones
                zones = ", ".join(data)
                lines.append(f"- {dev}: zones: {zones}")
            elif isinstance(data, dict):
                # Old format: data is dict with project_count and zones
                zones = ", ".join(data.get("zones", []))
                lines.append(f"- {dev}: {data.get('project_count', 0)} projects across zones: {zones}")

    # Diversified portfolios - handles both formats:
    # New format: list of {developer, technologies, project_count}
    # Old format: dict of {dev: {project_count, fuel_types}}
    diversified = dev_analysis.get("diversified_portfolios", [])
    if diversified:
        lines.append("\n**Diversified Portfolios (multiple fuel types):**")
        if isinstance(diversified, list):
            # New format: list of dicts
            for item in diversified[:5]:
                dev = item.get("developer", "Unknown")
                techs = ", ".join(item.get("technologies", item.get("fuel_types", [])))
                count = item.get("project_count", 0)
                lines.append(f"- {dev}: {count} projects ({techs})")
        elif isinstance(diversified, dict):
            # Old format: dict
            for dev, data in sorted(diversified.items(), key=lambda x: x[1].get("project_count", 0), reverse=True)[:5]:
                fuel_types = ", ".join(data.get("fuel_types", []))
                lines.append(f"- {dev}: {data.get('project_count', 0)} projects ({fuel_types})")

    return "\n".join(lines)


def get_analytics_context(analytics: Dict) -> str:
    """Format all analytics into context string for LLM."""
    lines = []

    # Summary stats - handles both "summary" and "corpus_stats" keys
    summary = analytics.get("summary") or analytics.get("corpus_stats", {})
    if summary:
        lines.append("### CORPUS SUMMARY")
        lines.append(f"- Total projects: {summary.get('total_projects', 'N/A')}")
        # Handle both key names for security data count
        sec_count = summary.get('projects_with_security') or summary.get('projects_with_security_data', 'N/A')
        lines.append(f"- Projects with security data: {sec_count}")

        sec_stats = summary.get("security_per_kw", {})
        if sec_stats:
            lines.append(f"- **Median security cost: ${sec_stats.get('median', 'N/A')}/kW** (n={sec_stats.get('n', 'N/A')})")
            lines.append(f"- Mean: ${sec_stats.get('mean', 'N/A')}/kW, Range: ${sec_stats.get('min', 'N/A')} - ${sec_stats.get('max', 'N/A')}/kW")

    # Fuel type breakdown
    fuel_stats = analytics.get("by_fuel_type", {})
    if fuel_stats:
        lines.append("\n### BY FUEL TYPE")
        lines.append(format_fuel_stats(fuel_stats))

    # Zone breakdown
    zone_stats = analytics.get("by_zone", {})
    if zone_stats:
        lines.append("\n### BY ZONE")
        lines.append(format_zone_stats(zone_stats))

    # TSP rankings
    tsp_rankings = analytics.get("tsp_rankings", [])
    if tsp_rankings:
        lines.append("\n### TSP RANKINGS (by avg $/kW)")
        lines.append(format_tsp_rankings(tsp_rankings))

    # Developer analysis
    dev_analysis = analytics.get("developer_analysis", {})
    if dev_analysis:
        lines.append("\n### DEVELOPER ANALYSIS")
        lines.append(format_developer_stats(dev_analysis))

    # Specific developer sections - handles both old format and new "specific_developers" format
    specific_devs = analytics.get("specific_developers", {})

    rwe_data = analytics.get("rwe_specific") or specific_devs.get("RWE", {})
    if rwe_data:
        lines.append("\n### RWE PROJECTS")
        lines.append(f"- Project count: {rwe_data.get('project_count', 'N/A')}")
        if rwe_data.get("median_security_per_kw"):
            lines.append(f"- Median security: ${rwe_data.get('median_security_per_kw')}/kW")
            vs_market = rwe_data.get("vs_market_median")
            if vs_market:
                lines.append(f"- vs Market median: {vs_market}")

    nextera_data = analytics.get("nextera_specific") or specific_devs.get("NEXTERA", {})
    if nextera_data:
        lines.append("\n### NEXTERA PROJECTS")
        lines.append(f"- Project count: {nextera_data.get('project_count', 'N/A')}")
        if nextera_data.get("median_security_per_kw"):
            lines.append(f"- Median security: ${nextera_data.get('median_security_per_kw')}/kW")

    # Data quality notes - handles both "low_sample_categories" and "low_sample_warnings" formats
    quality = analytics.get("data_quality", {})
    low_sample = quality.get("low_sample_categories") or quality.get("low_sample_warnings", [])
    if low_sample:
        lines.append("\n### DATA QUALITY NOTES")
        lines.append("Categories with limited sample size (n<10):")
        for item in low_sample[:5]:
            if isinstance(item, str):
                lines.append(f"- {item}")
            elif isinstance(item, dict):
                # New format: {category, bucket, sample_size, warning}
                cat = item.get("category", "")
                bucket = item.get("bucket", "")
                n = item.get("sample_size", "?")
                lines.append(f"- {cat}/{bucket}: n={n}")

    return "\n".join(lines)


def get_context_for_query(
    query: str,
    query_type: str,
    analytics: Optional[Dict],
    retrieval: Dict
) -> str:
    """Build context string based on query type."""
    if query_type == "aggregation" and analytics:
        return get_analytics_context(analytics)
    elif query_type == "hybrid" and analytics:
        return f"""## CORPUS-WIDE STATISTICS
{get_analytics_context(analytics)}

## RELEVANT DOCUMENT EXCERPTS
{retrieval.get('context', '')}
"""
    else:
        return retrieval.get("context", "")


# =============================================================================
# BUILT-IN COMPARATIVE QUERY FILTER HOOK
# Implements hard filtering, deduplication, and missing entity warnings
# =============================================================================

def create_comparative_filter_hook(
    k_total: int = None,
    max_chunks_per_project: int = 3  # Reduced from 5 to prevent LLM listing same project multiple times (Q20)
) -> Callable:
    """
    Create a filter hook that implements:
    - Multi-value filter extraction for comparative queries
    - Hard ChromaDB filtering with alias expansion
    - Deduplication by INR
    - Missing entity warnings

    Returns:
        Callable with signature (query, retriever) -> (docs, retrieval_dict, warning_or_none)
    """

    def comparative_filter_hook(query: str, retriever) -> Tuple[List, Dict, Optional[str]]:
        logger = get_logger()

        # Extract multi-value filters
        filters = extract_multi_filters_from_query(query)
        where_clause = build_chromadb_where_clause(filters, expand_aliases=True)

        # Determine if this is a comparative query that should use hard filtering
        # NOTE: Only hard filter on parent_company and tsp_normalized, NOT fuel_type
        # fuel_type has too many null values in the corpus, causing "No documents found" errors
        # Also trigger hard filtering for specific INR lookups (single project queries)
        is_comparative = (
            isinstance(filters.get('parent_company'), list) or
            isinstance(filters.get('tsp_normalized'), list)
        )

        # Special case: Single INR lookup should also use hard filtering for precision
        has_specific_inr = isinstance(filters.get('inr'), str) and filters.get('inr')

        # Log filter info
        if filters:
            logger.info(f"Extracted filters: {filters}")
            if where_clause:
                logger.info(f"ChromaDB where clause: {where_clause}")

        # Retrieve with hard filtering for:
        # 1. Comparative queries (RWE vs SAMSUNG, ONCOR vs Centerpoint)
        # 2. Specific INR lookups (25INR0138)
        # NOTE: Don't hard filter on fuel_type alone - too many null values cause empty results
        should_hard_filter = (is_comparative or has_specific_inr) and where_clause and hasattr(retriever, 'search_with_hard_filters')

        # Build a filtered where clause that excludes fuel_type from hard filtering
        # to prevent "No documents found" for battery vs solar comparisons
        hard_filter_clause = None

        # Check for zone-specific queries (Q6: "List all battery projects in West Texas zone")
        has_specific_zone = isinstance(filters.get('zone'), str) and filters.get('zone')

        # Also trigger hard filtering for zone-specific queries
        if has_specific_zone and not should_hard_filter:
            should_hard_filter = hasattr(retriever, 'search_with_hard_filters')

        if should_hard_filter:
            # Create a where clause with safe fields (parent_company, tsp_normalized, inr, zone)
            # NOTE: zone is safe because it has good coverage in metadata
            safe_filters = {k: v for k, v in filters.items()
                          if k in ('parent_company', 'tsp_normalized', 'inr', 'zone')}
            if safe_filters:
                hard_filter_clause = build_chromadb_where_clause(safe_filters, expand_aliases=True)

        if hard_filter_clause and hasattr(retriever, 'search_with_hard_filters'):
            filter_type = "INR lookup" if has_specific_inr else ("zone filter" if has_specific_zone else "comparative query")
            logger.info(f"Using HARD filtering mode for {filter_type}")
            docs = retriever.search_with_hard_filters(
                query,
                where=hard_filter_clause,
                k=k_total
            )
        else:
            docs = retriever.invoke(query)

        # Deduplication
        original_count = len(docs)
        docs = deduplicate_docs_by_inr(docs, max_chunks_per_project=max_chunks_per_project)
        if len(docs) < original_count:
            logger.info(f"Deduplicated: {original_count} -> {len(docs)} documents")

        # === POST-RETRIEVAL THRESHOLD FILTERING (Critical for Q2/Q19) ===
        # Filter out documents that don't meet threshold criteria BEFORE LLM sees them
        security_threshold = filters.get('security_per_kw_min')
        if security_threshold:
            pre_filter_count = len(docs)
            filtered_docs = []
            for doc in docs:
                doc_security = doc.metadata.get('security_per_kw')
                if doc_security is not None and doc_security >= security_threshold:
                    filtered_docs.append(doc)
                # Skip docs without security_per_kw for threshold queries

            if filtered_docs:
                docs = filtered_docs
                logger.success(f"Threshold filter (>=${security_threshold}/kW): {pre_filter_count} -> {len(docs)} docs")
            else:
                # If filter removed ALL docs, warn user
                logger.warning(f"Threshold filter removed all docs! Keeping original {pre_filter_count} docs with warning")

        # Check for missing entities (only for parent_company and tsp_normalized comparisons)
        # NOTE: Skip missing entity warnings for fuel_type - those comparisons work differently
        missing_warning = None
        if is_comparative:
            requested_entities = []
            entity_type = 'parent_company'  # Default

            if isinstance(filters.get('parent_company'), list):
                requested_entities = filters['parent_company']
                entity_type = 'parent_company'
            elif isinstance(filters.get('tsp_normalized'), list):
                requested_entities = filters['tsp_normalized']
                entity_type = 'tsp_normalized'
            # NOTE: Do NOT check missing entities for fuel_type comparisons (battery vs solar)
            # Those queries use semantic search and fuel_type metadata may be missing

            if requested_entities:
                # Use entity-type-aware functions for proper TSP vs parent company detection
                found_entities = get_entities_in_docs(docs, entity_type)
                missing = check_missing_entities(requested_entities, docs, entity_type)
                if missing:
                    lang = detect_language(query)
                    missing_warning = generate_attribution_warning(
                        missing, found_entities, lang
                    )
                    logger.warning(f"Missing entities in results: {missing}")

        # Format sources
        retrieval = format_sources(docs, max_sources=k_total)

        return docs, retrieval, missing_warning

    return comparative_filter_hook


# =============================================================================
# EXECUTE RETRIEVAL - Central routing with filter_hook extension point
# =============================================================================

def execute_retrieval(
    query: str,
    retriever,
    query_type: str,
    analytics_path: str,
    k_total: Optional[int] = None,
    filter_hook: Optional[Callable] = None
) -> Tuple[str, List, Dict, Optional[str]]:
    """
    Execute retrieval based on query type with optional filter hook.

    This function provides a clean extension point for hallucination guard
    and hard filter integration.

    Args:
        query: The user's question
        retriever: Document retriever
        query_type: "aggregation", "retrieval", or "hybrid"
        analytics_path: Path to analytics JSON
        k_total: Max documents to retrieve
        filter_hook: Optional callable(query, retriever) -> (docs, retrieval_dict, warning)
                    If not provided, uses built-in comparative_filter_hook

    Returns:
        Tuple of (context, docs, retrieval_dict, warning_or_none)
    """
    logger = get_logger()

    # --- AGGREGATION PATH: Skip retrieval entirely ---
    if query_type == "aggregation":
        analytics = load_analytics(analytics_path)
        if analytics:
            context = get_analytics_context(analytics)
            logger.success("Using pre-computed corpus analytics")
            retrieval = {
                "context": context,
                "has_docs": True,
                "sources": []
            }
            return context, [], retrieval, None
        else:
            logger.warning("Analytics unavailable, falling back to retrieval")
            query_type = "retrieval"  # Fallback

    # --- RETRIEVAL PATH: Apply filter hook ---
    docs = []
    missing_warning = None

    # Use built-in filter hook if none provided
    if filter_hook is None:
        filter_hook = create_comparative_filter_hook(k_total=k_total or config.K_DOCS_DEFAULT)

    # Execute filter hook
    docs, retrieval, missing_warning = filter_hook(query, retriever)

    if not docs and not retrieval.get("has_docs", True):
        # Empty retrieval
        pass

    # --- HYBRID PATH: Merge analytics with retrieval ---
    if query_type == "hybrid":
        analytics = load_analytics(analytics_path)
        if analytics:
            analytics_context = get_analytics_context(analytics)
            context = f"""## CORPUS-WIDE STATISTICS
{analytics_context}

## RELEVANT DOCUMENT EXCERPTS
{retrieval['context']}
"""
            retrieval["context"] = context
            logger.success("Using hybrid mode: analytics + document retrieval")
        else:
            context = retrieval["context"]
            logger.warning("Analytics unavailable for hybrid mode")
    else:
        context = retrieval["context"]
        logger.success("Using document retrieval with hard filtering")

    # --- THRESHOLD QUERY ENHANCEMENT (Critical for Q2/Q19 >$100/kW queries) ---
    # For threshold queries, get ALL projects meeting the threshold from ChromaDB
    filters = extract_multi_filters_from_query(query)
    security_threshold = filters.get('security_per_kw_min')
    if security_threshold and hasattr(retriever, 'get_all_projects_by_threshold'):
        logger.info(f"THRESHOLD QUERY: Getting all projects with security_per_kw >= ${security_threshold}/kW")
        try:
            threshold_projects = retriever.get_all_projects_by_threshold(
                threshold_field='security_per_kw',
                threshold_value=security_threshold,
                operator='$gte'
            )
            if threshold_projects:
                logger.success(f"Found {len(threshold_projects)} projects meeting threshold")
                # Format as a list to prepend to context
                threshold_list = f"\n## ALL PROJECTS WITH SECURITY >= ${security_threshold}/kW (Complete list from database)\n"
                threshold_list += f"Total: {len(threshold_projects)} projects\n\n"
                for i, proj in enumerate(threshold_projects, 1):
                    sec_val = proj.get('security_per_kw', 'N/A')
                    sec_str = f"${sec_val:.2f}/kW" if isinstance(sec_val, (int, float)) else sec_val
                    threshold_list += f"{i}. **{proj['project_name']}** ({proj['inr']}) - {sec_str}\n"
                    threshold_list += f"   Developer: {proj['developer']} | Zone: {proj['zone']} | TSP: {proj['tsp']}\n"
                # Prepend threshold list to context
                context = f"{threshold_list}\n{context}"
                retrieval["context"] = context
        except Exception as e:
            logger.warning(f"Threshold query failed: {e}")

    return context, docs, retrieval, missing_warning


# =============================================================================
# CHAIN BUILDERS
# =============================================================================

def get_flash_chain(
    retriever,
    k_total: int = None,
    with_history: bool = True,
    with_summary: bool = False,
    analytics_path: str = DEFAULT_ANALYTICS_PATH,
    filter_hook: Optional[Callable] = None
):
    """Build the FLASH mode RAG chain (analytics-aware with hard filtering).

    Args:
        retriever: Document retriever
        k_total: Max total documents to retrieve
        with_history: Whether to include chat history management
        with_summary: Whether to append document summary
        analytics_path: Path to pre-computed analytics JSON
        filter_hook: Optional callable for custom retrieval logic
                    Signature: (query, retriever) -> (docs, retrieval_dict, warning)

    Returns:
        RAG chain runnable (with or without history wrapper)
    """
    get_logger().info("Building FLASH mode chain (analytics-aware with hard filtering)")

    def flash_with_domain_filter(input_dict: Dict) -> Generator[str, None, None]:
        """Flash generator with domain pre-filter and analytics routing.

        Pipeline flow:
        1. Domain filter (out-of-scope rejection)
        2. Query classification (aggregation vs retrieval vs hybrid)
        3. Aggregation path → analytics JSON (skip retrieval)
        4. Retrieval path → filter_hook → hard filtering → deduplication
        5. Hybrid path → both analytics + retrieval
        6. Response generation
        """
        logger = get_logger()
        question = input_dict["question"]
        history = input_dict.get("chat_history", [])
        lang = detect_language(question)

        # --- STEP 1: Domain pre-filter ---
        if not is_domain_relevant(question, history):
            msg = OOS_QUESTION_MSG[lang]
            yield msg
            return

        # --- STEP 2: Query classification ---
        query_type = classify_query(question)
        logger.info(f"Query classified as: {query_type}")

        # --- STEPS 3-5: Execute retrieval based on query type ---
        context, docs, retrieval, missing_warning = execute_retrieval(
            query=question,
            retriever=retriever,
            query_type=query_type,
            analytics_path=analytics_path,
            k_total=k_total,
            filter_hook=filter_hook
        )

        # Check if we got any documents for retrieval queries
        if not retrieval.get("has_docs", True) and query_type != "aggregation":
            no_docs_msg = ("No tengo información sobre eso en los documentos disponibles."
                          if lang == 'spanish'
                          else "I don't have information about that in the available documents.")
            yield no_docs_msg
            return

        # --- Prepend missing entity warning if applicable ---
        if missing_warning:
            yield missing_warning
            yield "\n\n"

        # --- STEP 6: Generate response ---
        for chunk in generate_flash_response({
            "question": question,
            "retrieval": retrieval,
            "chat_history": history,
            "with_summary": with_summary,
            "_query_type": query_type
        }):
            yield chunk

    rag_chain_core = (
        RunnablePassthrough.assign(
            question=RunnableLambda(contextualize_question)
        )
        | RunnableLambda(flash_with_domain_filter)
    )

    if with_history:
        return RunnableWithMessageHistory(
            rag_chain_core,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
    return rag_chain_core


def get_thinking_chain(
    retriever,
    k_total: int = None,
    with_history: bool = True,
    with_summary: bool = False,
    analytics_path: str = DEFAULT_ANALYTICS_PATH,
    filter_hook: Optional[Callable] = None
):
    """Build the THINKING mode RAG chain (analytics-aware with hard filtering).

    Args:
        retriever: Document retriever
        k_total: Max total documents to retrieve across all queries
        with_history: Whether to include chat history management
        with_summary: Whether to append document summary
        analytics_path: Path to pre-computed analytics JSON
        filter_hook: Optional callable for custom retrieval logic
    """
    get_logger().info("Building THINKING mode chain (analytics-aware with hard filtering)")

    def thinking_generator(input_iter):
        """Generator function for RunnableGenerator - yields chunks from thinking response.

        Pipeline flow:
        1. Domain filter (out-of-scope rejection)
        2. Query classification (aggregation vs retrieval vs hybrid)
        3. Question contextualization (chat history)
        4. Thinking response generation (multi-step verification)
        """
        logger = get_logger()
        for input_dict in input_iter:
            question = input_dict.get("question", "")
            lang = detect_language(question)

            # --- STEP 1: Domain guardrail ---
            history = input_dict.get("chat_history", [])
            if not is_domain_relevant(question, history):
                msg = OOS_QUESTION_MSG[lang]
                yield msg
                return

            # --- STEP 2: Query classification ---
            query_type = classify_query(question)
            logger.info(f"Query classified as: {query_type}")

            # --- STEP 3: Extract filters for thinking mode ---
            filters = extract_multi_filters_from_query(question)
            # NOTE: Only hard filter on parent_company and tsp_normalized, NOT fuel_type
            # fuel_type has too many null values causing empty results for "battery vs solar"
            is_comparative = (
                isinstance(filters.get('parent_company'), list) or
                isinstance(filters.get('tsp_normalized'), list)
            )

            if filters:
                logger.info(f"Extracted filters for thinking: {filters}")

            # --- STEP 4: Contextualize question ---
            if input_dict.get("chat_history"):
                logger.step("Reformulating question based on chat history...")
                prompt_val = REPHRASE_PROMPT.invoke(input_dict)
                question = call_llm_api_full(prompt_val.to_string())
                logger.success(f"Reformulated: {question[:50]}...")
                input_dict = {**input_dict, "question": question}

            # Pass options to thinking response
            input_dict = {
                **input_dict,
                "with_summary": with_summary,
                "extracted_filters": filters,
                "is_comparative": is_comparative,
                "_query_type": query_type,
                "_analytics_path": analytics_path,
                "_filter_hook": filter_hook
            }

            # --- STEP 5: Generate thinking response ---
            for chunk in generate_thinking_response(input_dict, retriever, k_total=k_total):
                yield chunk

    rag_chain_core = RunnableGenerator(thinking_generator)

    if with_history:
        return RunnableWithMessageHistory(
            rag_chain_core,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
    return rag_chain_core


def get_rag_chain(
    retriever,
    mode: RAGMode = RAGMode.FLASH,
    k_docs: int = None,
    with_history: bool = True,
    with_summary: bool = False,
    analytics_path: str = DEFAULT_ANALYTICS_PATH,
    filter_hook: Optional[Callable] = None
):
    """Factory function to get the appropriate RAG chain based on mode.

    Args:
        retriever: Document retriever (SmartRetriever with hard filtering support)
        mode: RAGMode.FLASH or RAGMode.THINKING
        k_docs: Max number of documents to retrieve
        with_history: Whether to include chat history management
        with_summary: Whether to append document summary
        analytics_path: Path to pre-computed analytics JSON
        filter_hook: Optional callable for custom retrieval logic

    Example filter_hook implementation:
        def custom_filter_hook(query, retriever):
            # Custom filtering logic
            docs = retriever.invoke(query)
            retrieval = format_sources(docs)
            warning = None  # or generate warning
            return docs, retrieval, warning

    Returns:
        Configured RAG chain
    """
    if mode == RAGMode.THINKING:
        return get_thinking_chain(
            retriever,
            k_total=k_docs,
            with_history=with_history,
            with_summary=with_summary,
            analytics_path=analytics_path,
            filter_hook=filter_hook
        )
    else:
        return get_flash_chain(
            retriever,
            k_total=k_docs,
            with_history=with_history,
            with_summary=with_summary,
            analytics_path=analytics_path,
            filter_hook=filter_hook
        )

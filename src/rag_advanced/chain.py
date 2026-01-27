# Chain builders for RAG pipeline

import json
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableGenerator
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.chat_history import get_session_history
from src.llm_client import call_llm_api_full
from .utils import (
    get_logger, detect_language, format_sources, RAGMode, config
)
from .prompts import REPHRASE_PROMPT
from .components import (
    is_domain_relevant, contextualize_question,
    generate_flash_response, generate_thinking_response,
    classify_query
)

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


# --- Analytics Functions ---

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

    # Multi-zone developers
    multi_zone = dev_analysis.get("multi_zone_developers", {})
    if multi_zone:
        lines.append("**Multi-Zone Developers:**")
        for dev, zones in list(multi_zone.items())[:10]:
            lines.append(f"- {dev}: {', '.join(zones)}")

    # Diversified portfolios
    diversified = dev_analysis.get("diversified_portfolios", [])
    if diversified:
        lines.append("\n**Diversified Portfolios (2+ technologies):**")
        for item in diversified[:10]:
            lines.append(
                f"- {item['developer']}: {', '.join(item['technologies'])} "
                f"({item['project_count']} projects)"
            )

    return "\n".join(lines)


def format_specific_developers(specific_devs: Dict) -> str:
    """Format specific developer statistics."""
    lines = []
    for dev, data in specific_devs.items():
        if data.get("median_security_per_kw") is not None:
            lines.append(
                f"**{dev}:**\n"
                f"  - Projects: {data['project_count']} ({data.get('projects_with_security_data', 0)} with security data)\n"
                f"  - Median $/kW: ${data['median_security_per_kw']} (n={data.get('n', 'N/A')})\n"
                f"  - vs Corpus Median: {data['vs_corpus_median']}\n"
                f"  - Assessment: {data['assessment']}"
            )
        else:
            lines.append(
                f"**{dev}:**\n"
                f"  - Projects: {data['project_count']}\n"
                f"  - Security data: Not available"
            )
    return "\n\n".join(lines)


def format_data_quality_warnings(data_quality: Dict) -> str:
    """Format data quality warnings for context."""
    warnings = data_quality.get("low_sample_warnings", [])
    if not warnings:
        return ""

    lines = ["**Data Quality Notes:**"]
    for w in warnings[:5]:  # Limit to 5 warnings
        lines.append(f"- {w['category']}/{w['bucket']}: {w['warning']}")
    return "\n".join(lines)


def get_analytics_context(analytics: Dict) -> str:
    """Format full analytics context for LLM."""
    corpus = analytics.get("corpus_stats", {})
    security = corpus.get("security_per_kw", {})

    context = f"""## CORPUS ANALYTICS (Pre-computed from {corpus.get('total_projects', 'N/A')} projects)

### Security Cost Statistics ($/kW)
- Total projects: {corpus.get('total_projects', 'N/A')}
- Projects with security data: {corpus.get('projects_with_security_data', 'N/A')} (n={security.get('n', 'N/A')})
- Median: ${security.get('median', 'N/A')}/kW
- Mean: ${security.get('mean', 'N/A')}/kW
- Range: ${security.get('min', 'N/A')} - ${security.get('max', 'N/A')}/kW
- Std Dev: ${security.get('std_dev', 'N/A')}/kW

### By Technology Type
{format_fuel_stats(analytics.get('by_fuel_type', {}))}

### By Zone
{format_zone_stats(analytics.get('by_zone', {}))}

### TSP Rankings (by avg $/kW)
{format_tsp_rankings(analytics.get('tsp_rankings', []))}

### Geographic Concentration (Top Counties)
{format_county_stats(analytics.get('geographic_concentration', []))}

### Developer Intelligence
{format_developer_stats(analytics.get('developer_analysis', {}))}

### Specific Developer Analysis
{format_specific_developers(analytics.get('specific_developers', {}))}

{format_data_quality_warnings(analytics.get('data_quality', {}))}
"""
    return context


def get_context_for_query(
    query: str,
    retriever,
    analytics_path: str = DEFAULT_ANALYTICS_PATH
) -> Tuple[str, List, str]:
    """
    Get appropriate context based on query classification.

    Returns:
        Tuple of (context_string, retrieved_docs, query_type)
    """
    logger = get_logger()
    query_type = classify_query(query)
    logger.info(f"Query classified as: {query_type}")

    if query_type == "aggregation":
        # Load pre-computed analytics only
        analytics = load_analytics(analytics_path)
        if analytics:
            context = get_analytics_context(analytics)
            logger.success("Using pre-computed corpus analytics")
            return context, [], query_type
        else:
            # Fallback to retrieval if analytics unavailable
            logger.warning("Analytics unavailable, falling back to retrieval")
            docs = retriever.invoke(query)
            retrieval = format_sources(docs)
            return retrieval["context"], docs, "retrieval"

    elif query_type == "hybrid":
        # Get both analytics AND retrieved chunks
        analytics = load_analytics(analytics_path)
        docs = retriever.invoke(query)
        retrieval = format_sources(docs)

        if analytics:
            analytics_context = get_analytics_context(analytics)
            context = f"""## CORPUS-WIDE STATISTICS
{analytics_context}

## RELEVANT DOCUMENT EXCERPTS
{retrieval['context']}
"""
            logger.success("Using hybrid mode: analytics + document retrieval")
        else:
            context = retrieval["context"]
            logger.warning("Analytics unavailable for hybrid mode, using retrieval only")

        return context, docs, query_type

    else:  # retrieval
        docs = retriever.invoke(query)
        retrieval = format_sources(docs)
        logger.success("Using document retrieval")
        return retrieval["context"], docs, query_type


# --- Chain Builders ---

def get_flash_chain(
    retriever,
    k_total: Optional[int] = None,
    with_history: bool = True,
    with_summary: bool = False,
    analytics_path: str = DEFAULT_ANALYTICS_PATH
) -> Union[RunnableWithMessageHistory, RunnableLambda]:
    """Build Flash mode RAG chain (fast, 2-4 LLM calls with decomposition).

    Args:
        retriever: Document retriever
        k_total: Max total documents to retrieve (passed implicitly to/from retriever)
        with_history: Whether to include chat history management
        with_summary: Whether to append document summary
        analytics_path: Path to pre-computed analytics JSON

    Returns:
        RAG chain runnable (with or without history wrapper)
    """
    get_logger().info("Building FLASH mode chain (analytics-aware)")

    def flash_with_domain_filter(input_dict: Dict) -> Generator[str, None, None]:
        """Flash generator with domain pre-filter and analytics routing."""
        logger = get_logger()
        question = input_dict["question"]
        history = input_dict.get("chat_history", [])
        lang = detect_language(question)

        # Domain pre-filter: skip retrieval for out-of-scope questions
        if not is_domain_relevant(question, history):
            msg = OOS_QUESTION_MSG[lang]
            yield msg
            return

        # Get context based on query classification (analytics vs retrieval vs hybrid)
        context, docs, query_type = get_context_for_query(question, retriever, analytics_path)

        # Store query_type in input_dict for downstream access
        input_dict["_query_type"] = query_type

        # Build retrieval dict for generate_flash_response
        if query_type == "aggregation":
            # For pure aggregation, use analytics context directly
            retrieval = {
                "context": context,
                "has_docs": True,  # Analytics counts as having data
                "sources": []
            }
        else:
            # For retrieval or hybrid, format sources normally
            retrieval = format_sources(docs, max_sources=k_total)
            if query_type == "hybrid":
                # Prepend analytics context to retrieval context
                retrieval["context"] = context

        # Generate response
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
    analytics_path: str = DEFAULT_ANALYTICS_PATH
):
    """Build Thinking mode RAG chain (deep verification, 5-10 LLM calls).

    Args:
        retriever: Document retriever
        k_total: Max total documents to retrieve across all queries
        with_history: Whether to include chat history management
        with_summary: Whether to append document summary
        analytics_path: Path to pre-computed analytics JSON
    """
    get_logger().info("Building THINKING mode chain (analytics-aware)")

    def thinking_generator(input_iter):
        """Generator function for RunnableGenerator - yields chunks from thinking response."""
        logger = get_logger()
        for input_dict in input_iter:
            question = input_dict.get("question", "")
            lang = detect_language(question)

            # Domain guardrail FIRST - before any LLM calls (with chat context)
            history = input_dict.get("chat_history", [])
            if not is_domain_relevant(question, history):
                msg = OOS_QUESTION_MSG[lang]
                yield msg
                return

            # Classify query for analytics routing
            query_type = classify_query(question)
            logger.info(f"Query classified as: {query_type}")

            # Contextualize question (only if relevant)
            if input_dict.get("chat_history"):
                logger.step("Reformulating question based on chat history...")
                prompt_val = REPHRASE_PROMPT.invoke(input_dict)
                question = call_llm_api_full(prompt_val.to_string())
                logger.success(f"Reformulated: {question[:50]}...")
                input_dict = {**input_dict, "question": question}

            # Pass summary option and query type to thinking response
            input_dict = {
                **input_dict,
                "with_summary": with_summary,
                "_query_type": query_type,
                "_analytics_path": analytics_path
            }

            # Generate thinking response
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
    k_total: int = None,
    with_history: bool = True,
    with_summary: bool = False,
    analytics_path: str = DEFAULT_ANALYTICS_PATH
):
    """Get RAG chain with specified mode.

    Args:
        retriever: The document retriever
        mode: RAGMode.FLASH or RAGMode.THINKING
        k_total: Max total documents to retrieve
        with_history: Whether to include chat history management
        with_summary: Whether to append document summary to responses
        analytics_path: Path to pre-computed analytics JSON
    """
    if mode == RAGMode.FLASH:
        return get_flash_chain(
            retriever,
            k_total=k_total,
            with_history=with_history,
            with_summary=with_summary,
            analytics_path=analytics_path
        )
    else:
        return get_thinking_chain(
            retriever,
            k_total=k_total,
            with_history=with_history,
            with_summary=with_summary,
            analytics_path=analytics_path
        )

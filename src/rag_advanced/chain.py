# Chain builders for RAG pipeline
#
# Enhanced with:
# - Hard filtering for comparative queries (multi-value $in with alias expansion)
# - Deduplication by INR to prevent listing same projects multiple times
# - Attribution validation to catch hallucinations

from typing import Dict, Generator, Optional, Union
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableGenerator
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.chat_history import get_session_history
from src.llm_client import call_llm_api_full
from src.vector_store import (
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
    deduplicate_docs_by_inr, get_unique_projects_from_docs
)
from .attribution_validator import (
    validate_developer_attributions,
    check_missing_entities,
    generate_attribution_warning,
    get_developers_in_docs,
    create_grounding_context
)
from .alias_expander import get_canonical_parent

# Out-of-scope question messages
OOS_QUESTION_MSG = {
        "english": ("This question is not related to ERCOT interconnection agreements. "
                    "I can only answer questions about energy projects, power grids, and ERCOT."),
        "spanish": ("Esta pregunta no está relacionada con acuerdos de interconexión ERCOT. "
                    "Solo puedo responder preguntas sobre proyectos de energía, redes eléctricas y ERCOT.")
        }

# --- Chain Builders ---

def get_flash_chain(
    retriever,
    k_total: Optional[int] = None,
    with_history: bool = True,
    with_summary: bool = False
) -> Union[RunnableWithMessageHistory, RunnableLambda]:
    """Build Flash mode RAG chain (fast, 2-4 LLM calls with decomposition).

    Enhanced with:
    - Hard filtering for comparative queries (RWE vs SAMSUNG)
    - Alias expansion for parent companies and TSPs
    - Deduplication by INR to prevent listing same projects multiple times
    - Missing entity warnings for comparative queries

    Args:
        retriever: Document retriever
        k_total: Max total documents to retrieve (passed implicitly to/from retriever)
        with_history: Whether to include chat history management
        with_summary: Whether to append document summary

    Returns:
        RAG chain runnable (with or without history wrapper)
    """
    get_logger().info("Building FLASH mode chain (with hard filtering)")

    def flash_with_domain_filter(input_dict: Dict) -> Generator[str, None, None]:
        """Flash generator with domain pre-filter and enhanced retrieval."""
        logger = get_logger()
        question = input_dict["question"]
        history = input_dict.get("chat_history", [])
        lang = detect_language(question)

        # Domain pre-filter: skip retrieval for out-of-scope questions
        if not is_domain_relevant(question, history):
            msg = OOS_QUESTION_MSG[lang]
            yield msg
            return

        # === ENHANCED RETRIEVAL ===
        # Extract multi-value filters for comparative queries
        filters = extract_multi_filters_from_query(question)
        where_clause = build_chromadb_where_clause(filters, expand_aliases=True)

        # Determine if this is a comparative query
        is_comparative = (
            isinstance(filters.get('parent_company'), list) or
            isinstance(filters.get('tsp_normalized'), list) or
            isinstance(filters.get('fuel_type'), list)
        )

        # Log filter info
        if filters:
            logger.info(f"Extracted filters: {filters}")
            if where_clause:
                logger.info(f"ChromaDB where clause: {where_clause}")

        # Retrieve with hard filtering if comparative, otherwise use standard retrieval
        if where_clause and hasattr(retriever, 'search_with_hard_filters'):
            logger.info("Using HARD filtering mode for comparative query")
            docs = retriever.search_with_hard_filters(
                question,
                where=where_clause,
                k=k_total or 15
            )
        else:
            docs = retriever.invoke(question)

        # === DEDUPLICATION ===
        # Limit chunks per project to prevent repetition
        original_count = len(docs)
        docs = deduplicate_docs_by_inr(docs, max_chunks_per_project=2)
        if len(docs) < original_count:
            logger.info(f"Deduplicated: {original_count} -> {len(docs)} documents")

        # === CHECK FOR MISSING ENTITIES ===
        # For comparative queries, warn if some requested entities have no data
        missing_warning = None
        if is_comparative:
            requested_entities = []
            if isinstance(filters.get('parent_company'), list):
                requested_entities = filters['parent_company']
            elif isinstance(filters.get('tsp_normalized'), list):
                requested_entities = filters['tsp_normalized']

            if requested_entities:
                found_entities = get_developers_in_docs(docs)
                missing = check_missing_entities(requested_entities, docs)
                if missing:
                    missing_warning = generate_attribution_warning(
                        missing, found_entities, lang
                    )
                    logger.warning(f"Missing entities in results: {missing}")

        # Format sources
        retrieval = format_sources(docs, max_sources=k_total)

        # === GENERATE RESPONSE ===
        # Prepend missing entity warning if needed
        if missing_warning:
            yield missing_warning
            yield "\n\n"

        # Generate main response
        for chunk in generate_flash_response({
            "question": question,
            "retrieval": retrieval,
            "chat_history": history,
            "with_summary": with_summary
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


def get_thinking_chain(retriever, k_total: int = None, with_history: bool = True, with_summary: bool = False):
    """Build Thinking mode RAG chain (deep verification, 5-10 LLM calls).

    Enhanced with:
    - Hard filtering for comparative queries with alias expansion
    - Deduplication by INR
    - Missing entity warnings

    Args:
        retriever: Document retriever
        k_total: Max total documents to retrieve across all queries
        with_history: Whether to include chat history management
        with_summary: Whether to append document summary
    """
    get_logger().info("Building THINKING mode chain (with hard filtering)")

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

            # === CHECK FOR COMPARATIVE QUERY ===
            filters = extract_multi_filters_from_query(question)
            is_comparative = (
                isinstance(filters.get('parent_company'), list) or
                isinstance(filters.get('tsp_normalized'), list) or
                isinstance(filters.get('fuel_type'), list)
            )

            if filters:
                logger.info(f"Extracted filters for thinking: {filters}")

            # Contextualize question (only if relevant)
            if input_dict.get("chat_history"):
                logger.step("Reformulating question based on chat history...")
                prompt_val = REPHRASE_PROMPT.invoke(input_dict)
                question = call_llm_api_full(prompt_val.to_string())
                logger.success(f"Reformulated: {question[:50]}...")
                input_dict = {**input_dict, "question": question}

            # Pass summary option and filter info to thinking response
            input_dict = {
                **input_dict,
                "with_summary": with_summary,
                "extracted_filters": filters,
                "is_comparative": is_comparative
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


def get_rag_chain(retriever, mode: RAGMode = RAGMode.FLASH, k_total: int = None, with_history: bool = True, with_summary: bool = False):
    """Get RAG chain with specified mode.

    Args:
        retriever: The document retriever
        mode: RAGMode.FLASH or RAGMode.THINKING
        k_total: Max total documents to retrieve
        with_history: Whether to include chat history management
        with_summary: Whether to append document summary to responses
    """
    if mode == RAGMode.FLASH:
        return get_flash_chain(retriever, k_total=k_total, with_history=with_history, with_summary=with_summary)
    else:
        return get_thinking_chain(retriever, k_total=k_total, with_history=with_history, with_summary=with_summary)

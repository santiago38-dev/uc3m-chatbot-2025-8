# Chain builders for RAG pipeline

from typing import Dict, Generator, Optional, Union
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
    generate_flash_response, generate_thinking_response
)
from .query_filter_extractor import (
    extract_query_filters, format_filters_for_logging, validate_retrieved_docs
)
from .hallucination_guard import validate_entities_exist

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

    Args:
        retriever: Document retriever
        k_total: Max total documents to retrieve (passed implicitly to/from retriever)
        with_history: Whether to include chat history management
        with_summary: Whether to append document summary

    Returns:
        RAG chain runnable (with or without history wrapper)
    """
    get_logger().info("Building FLASH mode chain")

    def flash_with_domain_filter(input_dict: Dict) -> Generator[str, None, None]:
        """Flash generator with domain pre-filter, hard filtering, and hallucination guard."""
        logger = get_logger()
        question = input_dict["question"]
        history = input_dict.get("chat_history", [])
        lang = detect_language(question)

        # Domain pre-filter: skip retrieval for out-of-scope questions
        if not is_domain_relevant(question, history):
            msg = OOS_QUESTION_MSG[lang]
            yield msg
            return

        # Extract filters from query (numeric comparisons, zones, etc.)
        extracted_filters = extract_query_filters(question)
        where_clause = extracted_filters.to_chromadb_where()

        # Log what filters were extracted
        if not extracted_filters.is_empty():
            logger.info(f"Flash filters: {format_filters_for_logging(extracted_filters)}")

        # HALLUCINATION GUARD: Check if queried entities exist in corpus
        if extracted_filters.equality_filters:
            entity_check = validate_entities_exist(extracted_filters.equality_filters)
            if entity_check.get("should_abort", False):
                missing = entity_check.get("missing_entities", [])
                abort_msg = (
                    f"I don't have information about {', '.join(missing)} in my database of "
                    f"ERCOT interconnection agreements. Please check the spelling or try a different query."
                )
                yield abort_msg
                return

        # Retrieve documents with hard filtering if filters exist
        if where_clause and hasattr(retriever, 'search_with_hard_filters'):
            docs = retriever.search_with_hard_filters(question, where_clause)
            logger.info(f"Hard-filtered retrieval: {len(docs)} docs")

            # If hard filtering returns 0 docs, try relaxing numeric filters only
            if len(docs) == 0 and extracted_filters.numeric_filters:
                logger.info("No results with numeric filter, trying equality filters only")
                equality_only = {k: {"$eq": v} for k, v in extracted_filters.equality_filters.items()}
                if equality_only:
                    where_relaxed = {"$and": list(equality_only.values())} if len(equality_only) > 1 else list(equality_only.values())[0]
                    docs = retriever.search_with_hard_filters(question, where_relaxed)
                    logger.info(f"Relaxed filter retrieval: {len(docs)} docs")

            # If still no docs, fall back to boosted search but log warning
            if len(docs) == 0:
                logger.warning("Hard filtering returned 0 docs, falling back to boosted search")
                docs = retriever.search_with_filters(
                    question,
                    filters=extracted_filters.equality_filters
                )
        else:
            # No filters or retriever doesn't support hard filtering
            docs = retriever.invoke(question)

        # POST-RETRIEVAL VALIDATION: Check for filter leaks
        if not extracted_filters.is_empty():
            valid_docs, warnings = validate_retrieved_docs(docs, extracted_filters, logger)
            if warnings:
                logger.warning(f"Post-validation removed {len(docs) - len(valid_docs)} docs that violated filters")
                docs = valid_docs

        # Use k_total if explicitly passed, otherwise let retriever limit dictate
        retrieval = format_sources(docs, max_sources=k_total)

        # Generate response
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

    Args:
        retriever: Document retriever
        k_total: Max total documents to retrieve across all queries
        with_history: Whether to include chat history management
        with_summary: Whether to append document summary
    """
    get_logger().info("Building THINKING mode chain")

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

            # Contextualize question (only if relevant)
            if input_dict.get("chat_history"):
                logger.step("Reformulating question based on chat history...")
                prompt_val = REPHRASE_PROMPT.invoke(input_dict)
                question = call_llm_api_full(prompt_val.to_string())
                logger.success(f"Reformulated: {question[:50]}...")
                input_dict = {**input_dict, "question": question}

            # Pass summary option to thinking response
            input_dict = {
                **input_dict,
                "with_summary": with_summary
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

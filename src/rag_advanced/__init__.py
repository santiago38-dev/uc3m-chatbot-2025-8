"""
RAG Advanced Package - Lazy imports to avoid circular dependencies.

This package provides:
- Chain builders: get_flash_chain, get_thinking_chain, get_rag_chain
- Analytics: execute_retrieval, load_analytics, get_analytics_context, get_context_for_query, classify_query
- Filter utilities: extract_multi_filters_from_query, build_chromadb_where_clause
- Utilities: set_verbose, get_logger, RAGMode, config, QuestionType
- Session: get_session_history
"""

__all__ = [
    # Chain builders
    "get_flash_chain",
    "get_thinking_chain",
    "get_rag_chain",
    # Analytics functions
    "execute_retrieval",
    "load_analytics",
    "get_analytics_context",
    "get_context_for_query",
    "classify_query",
    # Filter utilities
    "extract_multi_filters_from_query",
    "build_chromadb_where_clause",
    # Utilities
    "set_verbose",
    "get_logger",
    "RAGMode",
    "config",
    "QuestionType",
    "get_session_history",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies and heavy loading."""

    # Chain functions - only load when actually needed (heavy deps: langchain)
    if name in ("get_flash_chain", "get_thinking_chain", "get_rag_chain"):
        from .chain import get_flash_chain, get_thinking_chain, get_rag_chain
        return {"get_flash_chain": get_flash_chain,
                "get_thinking_chain": get_thinking_chain,
                "get_rag_chain": get_rag_chain}[name]

    # Analytics functions from chain.py - also lazy due to dependencies
    if name in ("execute_retrieval", "load_analytics", "get_analytics_context", "get_context_for_query"):
        from .chain import execute_retrieval, load_analytics, get_analytics_context, get_context_for_query
        return {"execute_retrieval": execute_retrieval,
                "load_analytics": load_analytics,
                "get_analytics_context": get_analytics_context,
                "get_context_for_query": get_context_for_query}[name]

    # Query classification from components.py
    if name == "classify_query":
        from .components import classify_query
        return classify_query

    # Filter utilities - lightweight, no heavy dependencies
    if name in ("extract_multi_filters_from_query", "build_chromadb_where_clause"):
        from .filter_utils import extract_multi_filters_from_query, build_chromadb_where_clause
        return {"extract_multi_filters_from_query": extract_multi_filters_from_query,
                "build_chromadb_where_clause": build_chromadb_where_clause}[name]

    # Utils - lightweight, can load early
    if name in ("set_verbose", "get_logger", "RAGMode", "config", "QuestionType"):
        from .utils import set_verbose, get_logger, RAGMode, config, QuestionType
        return {"set_verbose": set_verbose,
                "get_logger": get_logger,
                "RAGMode": RAGMode,
                "config": config,
                "QuestionType": QuestionType}[name]

    # Session history
    if name == "get_session_history":
        from src.chat_history import get_session_history
        return get_session_history

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""
RAG Advanced Package - Lazy imports to avoid circular dependencies.

This package provides:
- get_flash_chain, get_thinking_chain, get_rag_chain (from chain.py)
- set_verbose, get_logger, RAGMode, config, QuestionType (from utils.py)
- get_session_history (from chat_history)
- filter_utils (extract_multi_filters_from_query, build_chromadb_where_clause)
"""

__all__ = [
    "get_flash_chain",
    "get_thinking_chain",
    "get_rag_chain",
    "set_verbose",
    "get_logger",
    "RAGMode",
    "config",
    "QuestionType",
    "get_session_history",
    "extract_multi_filters_from_query",
    "build_chromadb_where_clause",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies and heavy loading."""

    # Chain functions - only load when actually needed
    if name in ("get_flash_chain", "get_thinking_chain", "get_rag_chain"):
        from .chain import get_flash_chain, get_thinking_chain, get_rag_chain
        return {"get_flash_chain": get_flash_chain,
                "get_thinking_chain": get_thinking_chain,
                "get_rag_chain": get_rag_chain}[name]

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

    # Filter utilities - lightweight, no heavy dependencies
    if name in ("extract_multi_filters_from_query", "build_chromadb_where_clause"):
        from .filter_utils import extract_multi_filters_from_query, build_chromadb_where_clause
        return {"extract_multi_filters_from_query": extract_multi_filters_from_query,
                "build_chromadb_where_clause": build_chromadb_where_clause}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

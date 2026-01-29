# Utility functions, Enums, and Configuration for RAG pipeline

from enum import Enum
from typing import Dict, Any, List, Callable, Tuple
import re
import json
from langdetect import detect, LangDetectException

# --- Enums ---

class RAGMode(Enum):
    FLASH = "flash"
    THINKING = "thinking"

class QuestionType(Enum):
    """Question types for response format customization."""
    YES_NO = "yes_no"                # ¿Tiene NextEra proyectos en Texas? → SÍ/NO + justificación
    COMPARATIVE = "comparative"      # ¿Cómo se comparan los costos de NextEra vs RWE? → Tabla/lista comparativa
    AGGREGATION = "aggregation"      # ¿Cuál es el costo promedio por MW? → Valor agregado + desglose
    FACTUAL = "factual"             # ¿Cuál es el Security Amount del proyecto X? → Dato específico + contexto
    LISTING = "listing"             # ¿Qué proyectos tiene NextEra? → Lista estructurada
    TEMPORAL = "temporal"           # ¿Cómo han cambiado los costos desde 2020? → Análisis temporal/tendencias
    DEFINITIONAL = "definitional"   # ¿Qué es un SGIA? → Definición + contexto del corpus
    GENERAL = "general"             # Pregunta estándar sin formato especial

# =============================================================================
# CONFIGURATION - Adjust these parameters as needed
# =============================================================================

class RAGConfig:
    """Centralized configuration for RAG pipeline parameters."""

    # --- Document Retrieval ---
    K_DOCS_DEFAULT = 20          # Default docs to retrieve from vector store

    # --- Flash Mode ---
    FLASH_MAX_SOURCES = 10        # Max documents to include in Flash response

    # --- Thinking Mode ---
    THINKING_MAX_QUERIES = 6            # Max query variants for expansion (1 original + generated)

    # --- Parallelization ---
    RETRIEVAL_WORKERS = 4        # Parallel workers for multi-query retrieval

# Global config instance
config = RAGConfig()


# --- Precompiled Regex Patterns ---

# Patterns for meta-commentary removal (used in clean_response)
META_COMMENTARY_PATTERNS = [
    re.compile(r"(?i)^Note:.*$"),
    re.compile(r"(?i)^I have removed.*$"),
    re.compile(r"(?i)^I added a qualifier.*$"),
    re.compile(r"(?i)^I removed.*$"),
    re.compile(r"(?i)^This revised response.*$"),
    re.compile(r"(?i)^Based on the provided source documents, here is a revised.*$"),
    re.compile(r"(?i)^Here is a revised response.*$"),
    re.compile(r"(?i).*maintains accuracy with the source.*$"),
    re.compile(r"(?i).*cannot be verified against.*$"),
    re.compile(r"(?i).*as these statements cannot be verified.*$"),
]

# Pattern for duplicate source section removal
SOURCE_SECTION_PATTERN = re.compile(
    r'\n\n(?:Sources?|Fuentes?|References?):?\s*\n(?:\s*\[?\d+\]?[^\n]*\n?)*\s*$',
    re.IGNORECASE
)

# Pattern for consecutive newlines
CONSECUTIVE_NEWLINES_PATTERN = re.compile(r'\n{3,}')


# --- Verbose Logger ---

class VerboseLogger:
    """Handles verbose output for debugging/progress visibility."""

    def __init__(self, enabled: bool = True, callback: Callable[[str], None] = None):
        self.enabled = enabled
        self.callback = callback or print

    def log(self, step: str, message: str, emoji: str = ""):
        if self.enabled:
            prefix = f"{emoji} " if emoji else ""
            self.callback(f"{prefix}[{step}] {message}")

    def step(self, message: str):
        self.log("STEP", message, "🔄")

    def success(self, message: str):
        self.log("OK", message, "✓")

    def warning(self, message: str):
        self.log("WARN", message, "⚠")

    def info(self, message: str):
        self.log("INFO", message, "ℹ")


# Global logger instance (can be replaced)
import contextvars

# ContextVar to store logger per-thread/async context
_logger_context = contextvars.ContextVar("verbose_logger", default=None)

def set_verbose(enabled: bool, callback: Callable[[str], None] = None):
    """Enable/disable verbose mode for the current context."""
    logger = VerboseLogger(enabled=enabled, callback=callback)
    _logger_context.set(logger)

def get_logger() -> VerboseLogger:
    """Get logger for current context, or default disabled one."""
    logger = _logger_context.get()
    if logger is None:
        # Return a default disabled logger if not set
        return VerboseLogger(enabled=False)
    return logger


# --- Helper Functions ---

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return 'spanish' if lang == 'es' else 'english'
    except LangDetectException:
        return 'english'


def format_sources(docs, max_sources: int = None) -> Dict[str, Any]:
    """Returns dict with formatted context string and source metadata list.

    Args:
        docs: List of documents
        max_sources: Optional limit on number of sources (default: no limit)
    """
    if not docs:
        return {"context": "", "sources": [], "has_docs": False}

    # Apply source limit if specified
    if max_sources:
        docs = docs[:max_sources]

    formatted_parts = []
    source_list = []

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        project = meta.get('project_name', 'Unknown')
        inr = meta.get('inr', 'N/A')
        # Use section (actual section ID like article_1), fall back to section_type
        section = meta.get('section') or meta.get('section_type', 'N/A')

        formatted_parts.append(f"[Source {i}: {project} ({inr}) - {section}]\n{doc.page_content}\n")
        source_list.append({
            'ref': i,
            'project_name': project,
            'inr': inr,
            'section': section,
            'zone': meta.get('zone'),
            'fuel_type': meta.get('fuel_type')
        })

    return {
        "context": "\n".join(formatted_parts),
        "sources": source_list,
        "has_docs": True
    }


def format_citations(sources: list) -> str:
    if not sources:
        return ""
    lines = ["\n\nSources:"]
    for s in sources:
        lines.append(f"  [{s['ref']}] {s['project_name']} ({s['inr']}) - {s['section']}")
    return "\n".join(lines)


def clean_response(text: str) -> str:
    """Remove meta-commentary and duplicate source sections from LLM responses."""
    # Remove trailing source section using precompiled pattern
    text = SOURCE_SECTION_PATTERN.sub('', text)

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        is_meta = False
        line_stripped = line.strip()
        for pattern in META_COMMENTARY_PATTERNS:
            if pattern.match(line_stripped):
                is_meta = True
                break
        if not is_meta:
            cleaned_lines.append(line)

    # Remove consecutive empty lines using precompiled pattern
    result = '\n'.join(cleaned_lines)
    result = CONSECUTIVE_NEWLINES_PATTERN.sub('\n\n', result)
    return result.strip()




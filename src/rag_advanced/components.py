# Core logic components for RAG pipeline

from typing import Dict, Generator, Any, List, Callable, Tuple, Literal
from collections import defaultdict
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.llm_client import call_llm_api, call_llm_api_full
from .utils import (
    get_logger, detect_language, format_sources, format_citations, clean_response,
    QuestionType, RAGConfig, config
)
from .prompts import (
    DOMAIN_CHECK_PROMPT, REPHRASE_PROMPT, SUMMARY_PROMPT,
    QUESTION_TYPE_PROMPT, RESPONSE_VALIDATION_PROMPT,
    QUERY_EXPANSION_PROMPT, METADATA_EXTRACTION_PROMPT,
    SYSTEM_EN, SYSTEM_ES
)


# =============================================================================
# QUERY CLASSIFICATION FOR ANALYTICS ROUTING
# Routes queries to: aggregation (analytics JSON) | retrieval (ChromaDB) | hybrid
# =============================================================================

# Patterns that indicate aggregation queries (require corpus-wide statistics)
AGGREGATION_SIGNALS = [
    # Statistical keywords
    r'\b(median|average|mean|total|sum|count|range|distribution)\b',
    r'\b(rank|ranking|top|bottom|highest|lowest|most|least)\b',
    r'\b(compare|comparison|versus|vs)\b.*\b(corpus|market|industry|overall)\b',

    # Aggregation question patterns
    r'\bhow many\b',
    r'\bwhat percentage\b',
    r'\bwhat proportion\b',
    r'\bacross all\b',
    r'\bin the corpus\b',
    r'\boverall\b',
    r'\bmarket (average|median|rate)\b',

    # Specific aggregation questions
    r'\bwhich (developers?|companies|tsps?|counties|zones?|projects?) have\b',
    r'\bgeographic (concentration|distribution|patterns?)\b',
    r'\bdiversified (portfolio|mix)\b',
    r'\bmulti[- ]?zone\b',
    r'\b(typical|standard|normal) (security|cost|rate)\b',

    # ERCOT-specific aggregation
    r'\b(security costs?|security deposits?|security amounts?).*(per kw|per mw|\$/kw|\$/mw)\b',
    # Threshold queries on $/kW should use analytics (has pre-computed values for all projects)
    r'\b(over|above|greater|more than|>\s*)\$?\d+[/\\](kw|mw)\b',
    r'\bsecurity.*(over|above|greater|more than|>\s*)\$?\d+\b',
    r'\btsp.*(ranking|comparison|average|median)\b',
    r'\bzone.*(comparison|average|median|breakdown)\b',
]

# Patterns for specific entity mentions (suggests retrieval component needed)
ENTITY_PATTERNS = [
    r'\b\d{2}INR\d{4,5}\b',  # INR number format: YYINR####
    r'\bproject\s+[A-Z][a-zA-Z0-9\s]+\b',  # "Project Quantum", "Project Solar Farm"
    r'\bsgia\b',  # Specific agreement reference
    r'\b(samsung|intersect|rwe|nextera|invenergy|enel|pattern|terra-?gen|longroad|savion)\b',  # Known developers
    r'\b(oncor|centerpoint|aep|tnmp|lcra|austin energy)\b',  # Known TSPs (specific mention)
]


def classify_query(query: str) -> Literal["aggregation", "retrieval", "hybrid"]:
    """
    Classify query type to route to appropriate data source.

    Returns:
        "aggregation" - Use pre-computed analytics JSON (corpus-wide statistics)
        "retrieval" - Use ChromaDB semantic search (specific documents)
        "hybrid" - Use both (e.g., "Compare RWE to market" needs RWE chunks + market stats)

    Examples:
        "What's the median security cost per kW?" -> "aggregation"
        "What are the force majeure terms?" -> "retrieval"
        "Compare RWE to corpus median" -> "hybrid"
        "Which developers have projects in multiple zones?" -> "aggregation"
    """
    query_lower = query.lower()

    # Count aggregation signals
    aggregation_score = sum(
        1 for pattern in AGGREGATION_SIGNALS
        if re.search(pattern, query_lower)
    )

    # Check for specific entity mentions
    has_specific_entity = any(
        re.search(pattern, query, re.IGNORECASE)
        for pattern in ENTITY_PATTERNS
    )

    # Decision logic
    if aggregation_score >= 2 and not has_specific_entity:
        return "aggregation"
    elif aggregation_score >= 1 and has_specific_entity:
        return "hybrid"
    elif aggregation_score >= 1:
        # Single aggregation signal without entity - still likely aggregation
        return "aggregation"
    else:
        return "retrieval"


# Type alias for query classification result
AnalyticsQueryType = Literal["aggregation", "retrieval", "hybrid"]


# =============================================================================
# DOCUMENT DEDUPLICATION
# Prevents LLM from listing the same project multiple times
# =============================================================================

def deduplicate_docs_by_inr(
    docs: List[Document],
    max_chunks_per_project: int = 5
) -> List[Document]:
    """
    Limit chunks per project to prevent repetition in LLM output.

    Problem: When 15 chunks come from 3 projects (5 each), the LLM
    tends to list each project 5 times in the response.

    Solution: Keep only the most informative chunks per project INR.

    Args:
        docs: List of retrieved documents
        max_chunks_per_project: Maximum chunks to keep per unique INR

    Returns:
        Deduplicated list of documents
    """
    if not docs:
        return []

    # Group by INR
    inr_groups: Dict[str, List[Document]] = defaultdict(list)
    for doc in docs:
        inr = doc.metadata.get('inr', 'unknown')
        inr_groups[inr].append(doc)

    def score_chunk(doc: Document) -> int:
        """Score a chunk by informativeness (higher = better)."""
        score = 0
        meta = doc.metadata

        # Prefer chunks with key financial data
        if meta.get('security_amount'):
            score += 10
        if meta.get('security_per_kw'):
            score += 10
        if meta.get('nameplate_capacity_mw') or meta.get('capacity_mw'):
            score += 5

        # Prefer certain section types (Legal Evidence Bias)
        section = str(meta.get('section_type', '') or meta.get('section', '')).lower()
        if 'exhibit' in section:
            score += 10  # Exhibits contain the actual data (security amounts, costs)
        if 'schedule' in section:
            score += 8   # Schedules contain structured data tables
        if 'article' in section:
            score += 3   # Articles are usually boilerplate
        if 'annex' in section:
            score += 3

        # Prefer longer content (more context)
        content = doc.page_content or ''
        if len(content) > 1000:
            score += 3
        elif len(content) > 500:
            score += 2

        return score

    # Select top chunks per project
    result = []
    for inr, chunks in inr_groups.items():
        sorted_chunks = sorted(chunks, key=score_chunk, reverse=True)
        result.extend(sorted_chunks[:max_chunks_per_project])

    return result


def get_unique_projects_from_docs(docs: List[Document]) -> Dict[str, Dict[str, Any]]:
    """
    Extract unique projects from documents for listing purposes.

    Args:
        docs: List of documents

    Returns:
        Dict mapping INR to project info
        Example: {'25INR0138': {'name': 'Champaign BESS', 'developer': 'SAMSUNG', ...}}
    """
    projects = {}

    for doc in docs:
        meta = doc.metadata
        inr = meta.get('inr', '')
        if not inr or inr in projects:
            continue

        projects[inr] = {
            'name': meta.get('project_name', ''),
            'inr': inr,
            'developer': meta.get('parent_company', '') or meta.get('developer_spv', ''),
            'fuel_type': meta.get('fuel_type', ''),
            'capacity_mw': meta.get('capacity_mw', 0),
            'zone': meta.get('zone', ''),
            'tsp': meta.get('tsp_normalized', ''),
        }

    return projects


# --- Domain Filter ---

def is_domain_relevant(question: str, chat_history: list = None, threshold: float = 0.40) -> bool:
    """Check if question is about ERCOT/energy domain using confidence score.

    Args:
        question: The user's question
        chat_history: Optional chat history for context (follow-up questions)
        threshold: Minimum confidence score to consider relevant (default: 40%)
                  Lowered from 50% to allow legal/contractual questions like force majeure
    """
    logger = get_logger()
    logger.step("Checking if question is in ERCOT domain...")

    # Build chat context string if history exists
    chat_context = ""
    if chat_history:
        # Get last few exchanges for context
        recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
        context_lines = []
        for msg in recent_history:
            role = getattr(msg, 'type', 'unknown')
            content = getattr(msg, 'content', str(msg))[:200]  # Limit length
            context_lines.append(f"{role}: {content}...")
        if context_lines:
            chat_context = "Recent conversation context:\n" + "\n".join(context_lines) + "\n\n"

    prompt = DOMAIN_CHECK_PROMPT.format(question=question, chat_context=chat_context)

    try:
        result = call_llm_api_full(prompt).strip()
        logger.info(f"Domain check raw response: {result[:50]}...")
        # Extract number from response
        import re
        numbers = re.findall(r'\d+', result)
        if numbers:
            confidence = int(numbers[0]) / 100.0
            # Cap at 1.0 if LLM returns >100
            confidence = min(confidence, 1.0)
        else:
            # Fallback: if "yes" in response, assume high confidence
            confidence = 0.9 if 'yes' in result.lower() else 0.1
    except Exception as e:
        logger.warning(f"Domain check failed: {e}, allowing question to proceed")
        return True  # Default to allowing the question on error

    is_relevant = confidence >= threshold
    if is_relevant:
        logger.success(f"Question is domain-relevant ({confidence:.0%} confidence)")
    else:
        logger.info(f"Question is OUT OF SCOPE ({confidence:.0%} < {threshold:.0%} threshold)")

    return is_relevant





def generate_summary(context: str, lang: str = 'english', max_docs: int = 3) -> str:
    """Generate summaries of the top retrieved documents (parallel calls)."""
    logger = get_logger()
    logger.step(f"Generating summary (top {max_docs} docs)...")

    # Extract individual documents from context
    import re
    # Split by [Source N: markers but keep the delimiters
    parts = re.split(r'(\[Source \d+:)', context)

    docs = []
    current_doc = ""

    # Reassemble parts into full documents (Marker + Content)
    for part in parts:
        if re.match(r'\[Source \d+:', part.strip()):
            if current_doc:
                docs.append(current_doc)
            current_doc = part
        else:
            current_doc += part

    if current_doc:
        docs.append(current_doc)

    # Filter out empty docs and ensure we have actual content
    docs = [d for d in docs if len(d.strip()) > 20]

    if not docs:
        logger.warning("No documents found to summarize")
        # Fallback: treat the whole context as one doc if it's short enough
        if context.strip():
            docs = [context[:3000]]
        else:
            return ""

    # Limit to max_docs
    docs_to_summarize = docs[:max_docs]

    # Summarize each document in parallel
    def summarize_single_doc(doc_content: str) -> str:
        """Summarize a single document."""
        prompt = SUMMARY_PROMPT.format(context=doc_content.strip())
        try:
            summary = call_llm_api_full(prompt).strip()
            # Clean any meta-commentary
            if summary.lower().startswith('note') or summary.lower().startswith('here'):
                lines = summary.split('\n')
                summary = '\n'.join(lines[1:]).strip()
            return summary
        except Exception as e:
            return f"(Error summarizing: {e})"

    summaries = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        summaries = list(executor.map(summarize_single_doc, docs_to_summarize))

    # Merge summaries with source headers
    result_lines = []
    for i, (doc, summary) in enumerate(zip(docs_to_summarize, summaries), 1):
        # Extract source name from doc line like "[Source 1: Project Name (INR) - section]"
        source_match = re.search(r'\[Source \d+:\s*(.*?)\]', doc)
        source_name = source_match.group(1).strip() if source_match else f"Source {i}"
        result_lines.append(f"**{source_name}**")
        result_lines.append("") # Force new paragraph
        result_lines.append(summary)
        result_lines.append("")  # Empty line between docs

    logger.success(f"Generated {len(summaries)} individual summaries")
    return '\n'.join(result_lines).strip()


# --- FLASH MODE COMPONENTS ---

def classify_question(question: str) -> QuestionType:
    """Classify question type for response format customization."""
    logger = get_logger()
    logger.step("Classifying question type...")

    prompt = QUESTION_TYPE_PROMPT.format(question=question)

    try:
        result = call_llm_api_full(prompt).strip().upper()
    except Exception as e:
        logger.warning(f"Question classification failed: {e}, defaulting to GENERAL")
        return QuestionType.GENERAL

    # Parse result - handle variations
    type_mapping = {
        "YES_NO": QuestionType.YES_NO,
        "YESNO": QuestionType.YES_NO,
        "YES/NO": QuestionType.YES_NO,
        "COMPARATIVE": QuestionType.COMPARATIVE,
        "COMPARISON": QuestionType.COMPARATIVE,
        "AGGREGATION": QuestionType.AGGREGATION,
        "AGGREGATE": QuestionType.AGGREGATION,
        "FACTUAL": QuestionType.FACTUAL,
        "FACT": QuestionType.FACTUAL,
        "LISTING": QuestionType.LISTING,
        "LIST": QuestionType.LISTING,
        "TEMPORAL": QuestionType.TEMPORAL,
        "TIME": QuestionType.TEMPORAL,
        "TREND": QuestionType.TEMPORAL,
        "DEFINITIONAL": QuestionType.DEFINITIONAL,
        "DEFINITION": QuestionType.DEFINITIONAL,
        "GENERAL": QuestionType.GENERAL,
    }

    # Try to match the result
    for key, qtype in type_mapping.items():
        if key in result:
            logger.success(f"Question type: {qtype.value}")
            return qtype

    # Default fallback
    logger.info("Question type: GENERAL (default)")
    return QuestionType.GENERAL


def get_format_instructions_helper(question_type: QuestionType, lang: str) -> str:
    # Wrapper around the imported one if logic needed, or just direct import
    from .utils import RESPONSE_FORMAT_TEMPLATES
    lang_key = "es" if lang == "spanish" else "en"
    template = RESPONSE_FORMAT_TEMPLATES.get(question_type.value.upper(), RESPONSE_FORMAT_TEMPLATES["GENERAL"])
    return template.get(lang_key, template["en"])

# Note: get_format_instructions is now imported from utils if I moved templates there?
# Wait, I put RESPONSE_FORMAT_TEMPLATES in prompts.py. So I need to import it.
# I imported get_format_instructions from utils in the imports above... wait, I didn't put get_format_instructions IN utils.
# I should have put it in utils or prompts.
# Let's check prompts.py... Yes, RESPONSE_FORMAT_TEMPLATES is in prompts.py.
# So I should define get_format_instructions HERE using the template from prompts.py.

def get_format_instructions(question_type: QuestionType, lang: str) -> str:
    from .prompts import RESPONSE_FORMAT_TEMPLATES
    lang_key = "es" if lang == "spanish" else "en"
    template = RESPONSE_FORMAT_TEMPLATES.get(question_type.value.upper(), RESPONSE_FORMAT_TEMPLATES["GENERAL"])
    return template.get(lang_key, template["en"])


def validate_response(question: str, question_type: QuestionType, response: str, context: str, lang: str) -> Tuple[str, bool]:
    """Validate response coherence and format compliance. Returns (response, was_fixed)."""
    logger = get_logger()
    logger.step("Validating response coherence and format...")

    prompt = RESPONSE_VALIDATION_PROMPT.format(
        question=question,
        question_type=question_type.value.upper(),
        response=response[:2000]  # Limit response length for validation
    )

    try:
        result = call_llm_api_full(prompt).strip()
    except Exception as e:
        logger.warning(f"Validation LLM call failed: {e}")
        return response, False

    # Try to parse JSON response
    try:
        # Clean up common JSON issues
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()

        validation = json.loads(result)
        is_coherent = validation.get("is_coherent", True)
        format_compliant = validation.get("format_compliant", True)
        issues = validation.get("issues", [])
        suggested_fix = validation.get("suggested_fix")

        if is_coherent and format_compliant:
            logger.success("Response validation: PASSED ✓")
            return response, False

        # Log issues
        logger.warning(f"Response validation: ISSUES FOUND")
        for issue in issues:
            logger.info(f"  - {issue}")

        # If there are issues, try to fix the response
        if suggested_fix and (not is_coherent or not format_compliant):
            logger.step("Attempting to fix response based on validation feedback...")

            fix_prompt = f"""Fix this response to address the following issues:

Original Question: {question}
Question Type: {question_type.value.upper()}
Original Response: {response}

Issues found:
{chr(10).join(f'- {issue}' for issue in issues)}

Suggested fix: {suggested_fix}

Source documents for reference:
{context[:2000]}

RULES:
- Address the specific issues mentioned
- Maintain source citations
- Follow the format expected for {question_type.value.upper()} questions
- Do NOT add meta-commentary about the fix

Fixed response:"""

            fixed_response = call_llm_api_full(fix_prompt)
            fixed_response = clean_response(fixed_response)  # Remove duplicate sources
            logger.success("Response fixed based on validation feedback")
            return fixed_response, True

        return response, False

    except json.JSONDecodeError:
        logger.info("Could not parse validation response, assuming OK")
        return response, False


def contextualize_question(input_dict: Dict) -> str:
    """Uses blocking call to reformulate question if history exists."""
    logger = get_logger()

    if not input_dict.get("chat_history"):
        return input_dict["question"]

    logger.step("Reformulating question based on chat history...")
    prompt_val = REPHRASE_PROMPT.invoke(input_dict)
    result = call_llm_api_full(prompt_val.to_string())
    logger.success(f"Reformulated: {result[:50]}...")
    return result


def generate_flash_response(input_dict: Dict) -> Generator[str, None, None]:
    """Flash mode: Direct response generation with minimal processing."""
    logger = get_logger()
    question = input_dict["question"]
    retrieval = input_dict["retrieval"]
    history = input_dict.get("chat_history", [])
    with_summary = input_dict.get("with_summary", False)

    # 1. Detect Language
    lang = detect_language(question)
    logger.info(f"Language detected: {lang}")

    # 2. Handle No Documents
    if not retrieval["has_docs"]:
        msg = ("No tengo información sobre eso en los documentos disponibles."
               if lang == 'spanish'
               else "I don't have information about that in the available documents.")
        yield msg
        return

    # 3. Classify Question Type
    question_type = classify_question(question)
    format_instructions = get_format_instructions(question_type, lang)

    # 4. Construct Prompt with format instructions
    context = retrieval["context"]
    system_template = SYSTEM_ES if lang == 'spanish' else SYSTEM_EN
    enhanced_system = f"{system_template}\n\n{format_instructions}"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", enhanced_system),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    prompt_str = prompt_template.invoke({
        "context": context,
        "chat_history": history,
        "question": question
    }).to_string()

    logger.step("Generating response...")

    # 5. Stream Response
    for token in call_llm_api(prompt_str):
        yield token

    logger.success("Response generated")

    # 6. Append Citations
    citations = format_citations(retrieval["sources"])
    if citations:
        yield citations

    # 7. Auto-summarization (optional)
    if with_summary:
        summary = generate_summary(context, lang)
        summary_header = "\n\n--- Resumen ---\n" if lang == 'spanish' else "\n\n--- Summary ---\n"
        yield summary_header
        yield summary


# --- THINKING MODE COMPONENTS ---

def expand_query(question: str) -> List[str]:
    """Generate multiple query variants for broader retrieval."""
    logger = get_logger()
    logger.step("Expanding query into multiple search variants...")

    prompt = QUERY_EXPANSION_PROMPT.format(question=question)
    result = call_llm_api_full(prompt)

    queries = [q.strip() for q in result.strip().split('\n') if q.strip()]
    # Use config value: THINKING_MAX_QUERIES includes original, so we take (N-1) variants
    max_variants = config.THINKING_MAX_QUERIES - 1  # Reserve 1 slot for original
    queries = [question] + queries[:max_variants]

    logger.success(f"Generated {len(queries)} query variants")
    for i, q in enumerate(queries):
        logger.info(f"  Query {i+1}: {q[:60]}...")

    return queries


def multi_retrieve(queries: List[str], retriever, filters: Dict[str, Any] = None, k: int = None) -> List:
    """Retrieve documents for multiple queries in parallel and merge results.

    Args:
        queries: List of query strings
        retriever: The retriever instance
        filters: Optional metadata filters to apply/boost
        k: Optional limit of documents per query
    """
    logger = get_logger()
    logger.step(f"Retrieving documents for {len(queries)} queries in parallel (k={k or 'auto'})...")

    all_docs = []
    seen_contents = set()

    def retrieve_single(query: str) -> List:
        # If retriever supports 'filters', pass them.
        # SmartRetriever uses invoke(query), but we can't easily pass kwargs via invoke()
        # unless we modify SmartRetriever to look for a special query dict or use a different method.
        # HACK: If filters exist, we can assume the retriever exposes a method to set them
        # or we rely on the retriever to handle it.
        # Given we are modifying this, we should assume the retriever is capable.

        # Determine internal K for this specific query if overridden
        # Standard retriever.invoke(q) doesn't accept k.
        # We need to rely on the retriever instance if it has search methods.

        current_retriever = retriever

        # If k is provided, we might need to adjust the retriever's k temporarily?
        # Or call similarity_search directly if it's exposed.
        # Assuming LangChain retriever, often vectorstore.as_retriever()

        # Using vectorstore directly if accessible (common pattern in custom retrievers)
        if hasattr(retriever, 'vectorstore'):
             # Use the underlying vectorstore search if possible to control k
             if filters:
                 # Check if our custom SmartRetriever
                 if hasattr(retriever, 'search_with_filters'):
                      # SmartRetriever search_with_filters usually respects self.k,
                      # we may need to hint k if possible, but our implementation currently doesn't allow overriding k in search_with_filters easily
                      # unless we changed it. Wait, I saw SmartRetriever definition earlier.
                      # It uses self.k.
                      pass

        # For now, let's just retrieve and slice manually if the retriever returns more

        docs = []
        if hasattr(retriever, 'search_with_filters'):
             docs = retriever.search_with_filters(query, filters)
        else:
             docs = retriever.invoke(query)

        # Enforce per-query k limit manually here incase retriever returned more
        if k and len(docs) > k:
            docs = docs[:k]

        return docs

    # Parallelize retrieval for all queries
    with ThreadPoolExecutor(max_workers=config.RETRIEVAL_WORKERS) as executor:
        futures = [executor.submit(retrieve_single, q) for q in queries]

        for future in as_completed(futures):
            docs = future.result()
            for doc in docs:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)

    logger.success(f"Retrieved {len(all_docs)} unique documents")
    return all_docs






# --- METADATA EXTRACTION ---

def extract_query_metadata(question: str) -> Dict[str, Any]:
    """Extract metadata filters from the question using LLM."""
    logger = get_logger()
    logger.step("Extracting metadata filters from question...")

    prompt = METADATA_EXTRACTION_PROMPT.format(question=question)

    try:
        result = call_llm_api_full(prompt).strip()

        # Robust JSON extraction
        # 1. Strip markdown code blocks if present
        cleaned_result = result
        if "```" in result:
             # Try to find the content inside ```json ... ``` or just ``` ... ```
             # We look for the first block
             import re
             code_block = re.search(r'```(?:json)?(.*?)```', result, re.DOTALL)
             if code_block:
                 cleaned_result = code_block.group(1).strip()

        # 2. Find the JSON object defined by { ... }
        # This regex matches the first '{' and greedily captures until the last '}'
        # We use dotall to capture newlines
        json_match = re.search(r'\{.*\}', cleaned_result, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            metadata = json.loads(json_str)
        else:
            # Fallback: try loading the cleaned string directly
            try:
                metadata = json.loads(cleaned_result)
            except json.JSONDecodeError:
                # RETRY STRATEGY: Common LLM error is using single quotes 'key': 'value'
                # We try to naively replace single quotes with double quotes if safe
                try:
                    # Simple heuristic: replace ' with "
                    # Note: this might break if content has apostrophes, but it's a last resort
                    fixed_result = cleaned_result.replace("'", '"')
                    metadata = json.loads(fixed_result)
                except Exception:
                   if not result:
                       logger.info("Metadata extraction returned empty response")
                       return {}
                   # Raise original error to show warning
                   raise json.JSONDecodeError("No JSON found (retry failed)", result, 0)

        # Filter out empty or null values
        metadata = {k: v for k, v in metadata.items() if v}

        if metadata:
            logger.success(f"Extracted metadata: {metadata}")
        else:
            logger.info("No metadata extracted (empty JSON)")

        return metadata
    except json.JSONDecodeError as e:
        logger.warning(f"Metadata extraction JSON failed: {e}. Raw content: {result[:100]}...")
        return {}
    except Exception as e:
        logger.warning(f"Metadata extraction failed: {e}")
        return {}


def generate_thinking_response(input_dict: Dict, retriever, k_total: int = None) -> Generator[str, None, None]:
    """Thinking mode: Structured response with validation.

    Enhanced with:
    - Analytics routing for aggregation queries (Q7-Q12)
    - Hard filtering for comparative queries with alias expansion
    - Post-retrieval threshold filtering for numeric queries (Q2/Q19)
    - Deduplication by INR to prevent listing same projects multiple times
    - Missing entity warnings

    Flow: Classify → Route (analytics/retrieval/hybrid) → Filter → Dedupe → Generate → Validate
    Note: Domain guardrail is checked in thinking_generator before this is called.
    """
    # Import from filter_utils to avoid circular imports
    from .filter_utils import build_chromadb_where_clause
    from .attribution_validator import (
        check_missing_entities,
        generate_attribution_warning,
        get_entities_in_docs
    )

    logger = get_logger()
    question = input_dict["question"]
    history = input_dict.get("chat_history", [])
    with_summary = input_dict.get("with_summary", False)
    extracted_filters = input_dict.get("extracted_filters", {})
    is_comparative = input_dict.get("is_comparative", False)

    # Get analytics routing info from chain.py
    query_type = input_dict.get("_query_type", "retrieval")
    analytics_path = input_dict.get("_analytics_path", "data/corpus_analytics.json")

    # Use k_total if provided, else fallback to config default
    max_docs = k_total if k_total else config.K_DOCS_DEFAULT

    # Detect language
    lang = detect_language(question)

    logger.info("=" * 50)
    logger.info("THINKING MODE ACTIVATED")
    logger.info("=" * 50)
    logger.info(f"Language: {lang}")
    logger.info(f"Query type: {query_type}")
    if is_comparative:
        logger.info(f"Comparative query detected with filters: {extracted_filters}")

    # === ANALYTICS ROUTING (Critical for Q7-Q12) ===
    # Load analytics for aggregation and hybrid queries
    analytics_context = None
    if query_type in ("aggregation", "hybrid"):
        try:
            from .chain import load_analytics, get_analytics_context
            analytics = load_analytics(analytics_path)
            if analytics:
                analytics_context = get_analytics_context(analytics)
                logger.success(f"Loaded corpus analytics for {query_type} query")
        except Exception as e:
            logger.warning(f"Failed to load analytics: {e}")

    # For pure aggregation queries, use analytics directly (skip retrieval)
    if query_type == "aggregation" and analytics_context:
        logger.info("AGGREGATION PATH: Using corpus analytics (no document retrieval)")

        # 2. Classify Question Type
        question_type = classify_question(question)
        format_instructions = get_format_instructions(question_type, lang)

        # Generate response from analytics
        system_template = SYSTEM_ES if lang == 'spanish' else SYSTEM_EN
        enhanced_system = f"{system_template}\n\n{format_instructions}"

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", enhanced_system),
            ("placeholder", "{chat_history}"),
            ("human", "{question}")
        ])

        prompt_str = prompt_template.invoke({
            "context": analytics_context,
            "chat_history": history,
            "question": question
        }).to_string()

        response = call_llm_api_full(prompt_str)
        response = clean_response(response)

        # Validation
        final_response, was_fixed = validate_response(
            question=question,
            question_type=question_type,
            response=response,
            context=analytics_context,
            lang=lang
        )

        # Stream output for aggregation
        validation_status = "Fixed ⚠" if was_fixed else "Passed ✓"
        thought_summary = f"""**💭 Analysis process:**
- Question type: {question_type.value.replace("_", " ").title()}
- Data source: Corpus Analytics (pre-computed statistics)
- Validation: {validation_status}

---

**📋 Answer:**
"""
        yield thought_summary
        yield final_response
        yield "\n\n**Sources:** Pre-computed corpus analytics from 132 ERCOT interconnection agreements"

        if with_summary:
            summary = generate_summary(analytics_context, lang)
            summary_header = "\n\n--- Resumen ---\n" if lang == 'spanish' else "\n\n--- Summary ---\n"
            yield summary_header
            yield summary
        return

    # 2. Classify Question Type (for retrieval/hybrid paths)
    question_type = classify_question(question)
    format_instructions = get_format_instructions(question_type, lang)

    # 3. Metadata Extraction (LLM-based, complements regex-based filters)
    metadata_filters = extract_query_metadata(question)

    # 3b. Also run regex-based extraction for threshold queries (more reliable for Q2/Q19)
    from .filter_utils import extract_multi_filters_from_query
    regex_filters = extract_multi_filters_from_query(question)
    # Merge threshold from regex extraction (more reliable than LLM for numeric thresholds)
    if regex_filters.get('security_per_kw_min') and not metadata_filters.get('security_per_kw_min'):
        metadata_filters['security_per_kw_min'] = regex_filters['security_per_kw_min']
        logger.info(f"Using regex-extracted threshold: ${regex_filters['security_per_kw_min']}/kW")

    # 4. Query Expansion
    queries = expand_query(question)

    # 5. Build hard filter for comparative, INR, zone, and project_name queries
    # NOTE: Only hard filter on parent_company, tsp_normalized, inr, zone, project_name
    # fuel_type has too many null values in the corpus, causing empty results
    where_clause = None
    has_specific_inr = isinstance(extracted_filters.get('inr'), str) and extracted_filters.get('inr')
    has_specific_zone = isinstance(extracted_filters.get('zone'), str) and extracted_filters.get('zone')

    # Check for project_name from LLM extraction (Q17: "What is the security deposit for Quantum Storage?")
    # BUT: Don't hard filter on project_name for comparative queries (Q13: "Compare Headcamp to Quantum")
    # because we need documents from BOTH projects, not just one
    has_specific_project = isinstance(metadata_filters.get('project_name'), str) and metadata_filters.get('project_name')

    # Detect if this is a project comparison query (mentions "compare", "vs", "versus" with project names)
    is_project_comparison = bool(re.search(r'\b(compare|vs\.?|versus)\b', question.lower()))

    # Merge project_name from LLM extraction into extracted_filters for hard filtering
    # ONLY for single-project lookups, NOT for project comparisons
    combined_filters = dict(extracted_filters) if extracted_filters else {}
    if has_specific_project and not is_project_comparison:
        combined_filters['project_name'] = metadata_filters['project_name']
        logger.info(f"LLM extracted project_name for hard filter: {metadata_filters['project_name']}")
    elif has_specific_project and is_project_comparison:
        logger.info(f"Skipping project_name hard filter for comparison query (need docs from multiple projects)")

    if combined_filters:
        # Build safe where clause excluding fuel_type from hard filtering
        # NOTE: zone and project_name are safe because they have good coverage in metadata
        safe_filters = {k: v for k, v in combined_filters.items()
                       if k in ('parent_company', 'tsp_normalized', 'inr', 'zone', 'project_name')}
        if safe_filters:
            where_clause = build_chromadb_where_clause(safe_filters, expand_aliases=True)
            if where_clause:
                logger.info(f"Built ChromaDB where clause (safe fields only): {where_clause}")

    # 6. Multi-Query Retrieval
    # Strategy: Split K total budget across N queries
    num_queries = len(queries)
    k_per_query = max(1, max_docs // num_queries) if max_docs else None

    logger.info(f"Retrieval strategy: {num_queries} queries, limit {k_per_query} docs per query (Total budget: {max_docs})")

    # Use hard filtering for comparative queries, INR lookups, zone-specific, or project-specific queries
    # NOTE: is_comparative is set in chain.py for parent_company and tsp_normalized comparisons
    # NOTE: Don't hard filter on project_name alone for project comparison queries (Q13)
    use_project_filter = has_specific_project and not is_project_comparison
    should_hard_filter = (is_comparative or has_specific_inr or has_specific_zone or use_project_filter) and where_clause and hasattr(retriever, 'search_with_hard_filters')

    if should_hard_filter:
        if has_specific_inr:
            filter_type = "INR lookup"
        elif use_project_filter:
            filter_type = "project name lookup"
        elif has_specific_zone:
            filter_type = "zone filter"
        else:
            filter_type = "comparative query"
        logger.info(f"Using HARD filtering mode for {filter_type}")
        # For hard filtering, we retrieve with the filter applied
        all_docs = retriever.search_with_hard_filters(question, where=where_clause, k=max_docs)
    else:
        # Standard multi-retrieve with boosting
        all_docs = multi_retrieve(queries, retriever, filters=metadata_filters, k=k_per_query)

    if not all_docs:
        msg = ("No tengo información sobre eso en los documentos disponibles."
               if lang == 'spanish'
               else "I don't have information about that in the available documents.")
        yield msg
        return

    # === DEDUPLICATION ===
    original_count = len(all_docs)
    all_docs = deduplicate_docs_by_inr(all_docs, max_chunks_per_project=3)  # Reduced from 5 to prevent LLM duplication (Q20)
    if len(all_docs) < original_count:
        logger.info(f"Deduplicated: {original_count} -> {len(all_docs)} documents")

    # === POST-RETRIEVAL THRESHOLD FILTERING (Critical for Q2/Q19) ===
    # Filter out documents that don't meet threshold criteria BEFORE LLM sees them
    # This prevents the LLM from hallucinating projects below the threshold
    security_threshold = metadata_filters.get('security_per_kw_min')
    if security_threshold:
        pre_filter_count = len(all_docs)
        filtered_docs = []
        for doc in all_docs:
            doc_security = doc.metadata.get('security_per_kw')
            if doc_security is not None and doc_security >= security_threshold:
                filtered_docs.append(doc)
            elif doc_security is None:
                # Keep docs without security data (might be relevant for other info)
                # but log this case
                logger.debug(f"Keeping doc without security_per_kw: {doc.metadata.get('inr', 'unknown')}")
                # Actually, for threshold queries, we should SKIP docs without the field
                # to avoid LLM confusion
                pass  # Skip docs without security_per_kw for threshold queries

        if filtered_docs:
            all_docs = filtered_docs
            logger.success(f"Threshold filter (>=${security_threshold}/kW): {pre_filter_count} -> {len(all_docs)} docs")
        else:
            # If filter removed ALL docs, warn but keep some for context
            logger.warning(f"Threshold filter removed all docs! Keeping original {pre_filter_count} docs with warning")
            # Add a flag to warn user
            if not missing_warning:
                missing_warning = f"⚠️ Note: Few or no projects in the retrieved documents meet the ${security_threshold}/kW threshold. Results may be limited.\n"

    # Final safety clamp to k_total (in case disjoint sets exceeded total)
    if max_docs and len(all_docs) > max_docs:
        logger.info(f"Clamping final merged documents from {len(all_docs)} to {max_docs}")
        all_docs = all_docs[:max_docs]

    # === CHECK FOR MISSING ENTITIES ===
    # Fixed: Use entity_type to correctly detect TSPs vs developers (fixes Q3 ONCOR warning bug)
    # NOTE: Only check for parent_company and tsp_normalized - NOT for fuel_type comparisons
    # fuel_type comparisons (battery vs solar) work via semantic search, not metadata matching
    missing_warning = None
    if is_comparative:
        requested_entities = []
        entity_type = 'parent_company'  # Default

        if isinstance(extracted_filters.get('parent_company'), list):
            requested_entities = extracted_filters['parent_company']
            entity_type = 'parent_company'
        elif isinstance(extracted_filters.get('tsp_normalized'), list):
            requested_entities = extracted_filters['tsp_normalized']
            entity_type = 'tsp_normalized'
        # NOTE: Do NOT check missing entities for fuel_type comparisons
        # Those queries work via semantic search and metadata may be incomplete

        if requested_entities:
            # Use entity-type-aware function for proper TSP vs developer detection
            found_entities = get_entities_in_docs(all_docs, entity_type)
            missing = check_missing_entities(requested_entities, all_docs, entity_type)
            if missing:
                missing_warning = generate_attribution_warning(missing, found_entities, lang)
                logger.warning(f"Missing {entity_type} entities in results: {missing}")

    # 7. Format sources for response
    retrieval = format_sources(all_docs)
    context = retrieval["context"]

    # === HYBRID CONTEXT MERGING (Critical for Q7-Q12 comparative analytics) ===
    # For hybrid queries, prepend analytics context to retrieved documents
    if query_type == "hybrid" and analytics_context:
        logger.info("HYBRID PATH: Merging corpus analytics with document retrieval")
        context = f"""## CORPUS-WIDE STATISTICS (Pre-computed from ALL 132 projects)
{analytics_context}

## RELEVANT DOCUMENT EXCERPTS (From specific matching projects)
{context}
"""
        retrieval["context"] = context

    # === THRESHOLD QUERY ENHANCEMENT (Critical for Q2/Q19 >$100/kW queries) ===
    # For threshold queries, get ALL projects meeting the threshold from ChromaDB
    # This ensures we don't miss projects that semantic search didn't find
    security_threshold = metadata_filters.get('security_per_kw_min')
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

    # 8. Generate Response
    logger.step("Generating response...")
    system_template = SYSTEM_ES if lang == 'spanish' else SYSTEM_EN
    enhanced_system = f"{system_template}\n\n{format_instructions}"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", enhanced_system),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    prompt_str = prompt_template.invoke({
        "context": context,
        "chat_history": history,
        "question": question
    }).to_string()

    response = call_llm_api_full(prompt_str)
    response = clean_response(response)
    logger.success("Response generated")

    # 7. Validate response format and citations
    final_response, was_fixed = validate_response(
        question=question,
        question_type=question_type,
        response=response,
        context=context,
        lang=lang
    )

    logger.info("=" * 50)
    logger.success("THINKING COMPLETE")
    logger.info("=" * 50)

    # 8. Build structured output
    question_type_label = question_type.value.replace("_", " ").title()
    validation_status = "Fixed ⚠" if was_fixed else "Passed ✓"

    if lang == 'spanish':
        thought_summary = f"""**💭 Proceso de análisis:**
- Tipo de pregunta: {question_type_label}
- Consultas generadas: {len(queries)}
- Documentos recuperados: {len(all_docs)}
- Validación: {"Corregida ⚠" if was_fixed else "Pasada ✓"}

---

**📋 Respuesta:**
"""
    else:
        thought_summary = f"""**💭 Analysis process:**
- Question type: {question_type_label}
- Queries generated: {len(queries)}
- Documents retrieved: {len(all_docs)}
- Validation: {validation_status}

---

**📋 Answer:**
"""

    # Stream output
    # Add missing entity warning first if applicable
    if missing_warning:
        yield missing_warning
        yield "\n\n"

    yield thought_summary
    yield final_response

    # 9. Add citations
    citations = format_citations(retrieval["sources"])
    if citations:
        yield citations

    # 10. Auto-summarization (optional)
    if with_summary:
        summary = generate_summary(context, lang)
        summary_header = "\n\n--- Resumen ---\n" if lang == 'spanish' else "\n\n--- Summary ---\n"
        yield summary_header
        yield summary

# Core logic components for RAG pipeline

from typing import Dict, Generator, Any, List, Callable, Tuple
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import ChatPromptTemplate

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
from .hallucination_guard import (
    validate_entities_exist,
    validate_response_grounding,
    create_grounding_warning
)
from .query_filter_extractor import (
    extract_query_filters,
    format_filters_for_logging,
    validate_retrieved_docs
)

# --- Domain Filter ---

def is_domain_relevant(question: str, chat_history: list = None, threshold: float = 0.50) -> bool:
    """Check if question is about ERCOT/energy domain using confidence score.

    Args:
        question: The user's question
        chat_history: Optional chat history for context (follow-up questions)
        threshold: Minimum confidence score to consider relevant (default: 50%)
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
    queries = [question] + queries[:3]  # Original + up to 3 variants

    logger.success(f"Generated {len(queries)} query variants")
    for i, q in enumerate(queries):
        logger.info(f"  Query {i+1}: {q[:60]}...")

    return queries


def multi_retrieve(
    queries: List[str],
    retriever,
    filters: Dict[str, Any] = None,
    hard_filters: Dict = None,
    k: int = None
) -> List:
    """Retrieve documents for multiple queries in parallel and merge results.

    Args:
        queries: List of query strings
        retriever: The retriever instance
        filters: Optional metadata filters to apply/boost (soft filtering)
        hard_filters: Optional ChromaDB where clause for hard pre-filtering
        k: Optional limit of documents per query
    """
    logger = get_logger()
    filter_mode = "hard" if hard_filters else ("boosted" if filters else "none")
    logger.step(f"Retrieving docs for {len(queries)} queries (k={k or 'auto'}, filter_mode={filter_mode})...")

    all_docs = []
    seen_contents = set()

    def retrieve_single(query: str) -> List:
        docs = []

        # Priority: Hard filters > Soft filters > No filters
        if hard_filters and hasattr(retriever, 'search_with_hard_filters'):
            docs = retriever.search_with_hard_filters(query, hard_filters)
        elif filters and hasattr(retriever, 'search_with_filters'):
            docs = retriever.search_with_filters(query, filters)
        else:
            docs = retriever.invoke(query)

        # Enforce per-query k limit
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

    Flow: Classify → Extract Metadata → Expand queries → Retrieve → Generate → Validate
    Note: Domain guardrail is checked in thinking_generator before this is called.
    """
    logger = get_logger()
    question = input_dict["question"]
    history = input_dict.get("chat_history", [])
    with_summary = input_dict.get("with_summary", False)

    # Use k_total if provided, else fallback to reasonable default or unlimited
    max_docs = k_total if k_total else 15

    # Detect language
    lang = detect_language(question)

    logger.info("=" * 50)
    logger.info("THINKING MODE ACTIVATED")
    logger.info("=" * 50)
    logger.info(f"Language: {lang}")

    # 2. Classify Question Type
    question_type = classify_question(question)
    format_instructions = get_format_instructions(question_type, lang)

    # 3. Metadata Extraction
    metadata_filters = extract_query_metadata(question)

    # 3.5 PRE-VALIDATION: Check if queried entities exist in corpus
    entity_warning = None
    if metadata_filters:
        entity_check = validate_entities_exist(metadata_filters)
        if not entity_check["valid"]:
            logger.warning(f"Entity validation failed: {entity_check['warning_message']}")
            entity_warning = create_grounding_warning(entity_check["missing_entities"])

            # ABORT if critical entity (like developer) is missing
            if entity_check.get("should_abort", False):
                logger.warning("Aborting query - critical entity not found in corpus")
                missing_info = entity_check["missing_entities"][0]  # Get first missing
                available_list = ", ".join(missing_info["available"][:8])

                if lang == 'spanish':
                    abort_msg = f"""**⚠️ Datos No Disponibles**

No se encontraron proyectos de **{missing_info['value']}** en el corpus.

**{missing_info['label'].title()}s disponibles:** {available_list}

Por favor reformule su pregunta usando uno de los {missing_info['label']}s disponibles."""
                else:
                    abort_msg = f"""**⚠️ Data Not Available**

No **{missing_info['value']}** projects found in the corpus.

**Available {missing_info['label']}s:** {available_list}

Please rephrase your question using one of the available {missing_info['label']}s."""

                yield abort_msg
                return

    # 4. Query Expansion
    queries = expand_query(question)

    # 4.5 Extract hard filters for ChromaDB where clause (same as Flash mode)
    extracted_filters = extract_query_filters(question)
    hard_where_clause = extracted_filters.to_chromadb_where()

    if not extracted_filters.is_empty():
        logger.info(f"Thinking filters: {format_filters_for_logging(extracted_filters)}")

    # 5. Multi-Query Retrieval with HARD FILTERING
    # Strategy: Split K total budget across N queries
    # Hard filters are applied to EXCLUDE non-matching docs (not just boost)
    num_queries = len(queries)
    k_per_query = max(1, max_docs // num_queries) if max_docs else None

    logger.info(f"Retrieval strategy: {num_queries} queries, limit {k_per_query} docs per query (Total budget: {max_docs})")

    all_docs = multi_retrieve(
        queries,
        retriever,
        filters=metadata_filters,  # Soft boosting from LLM extraction
        hard_filters=hard_where_clause,  # Hard pre-filtering from regex extraction
        k=k_per_query
    )

    if not all_docs:
        msg = ("No tengo información sobre eso en los documentos disponibles."
               if lang == 'spanish'
               else "I don't have information about that in the available documents.")
        yield msg
        return

    # 5.5 POST-RETRIEVAL VALIDATION: Remove any docs that violate filters
    if not extracted_filters.is_empty():
        valid_docs, filter_warnings = validate_retrieved_docs(all_docs, extracted_filters, logger)
        if filter_warnings:
            logger.warning(f"Post-validation removed {len(all_docs) - len(valid_docs)} docs that violated filters")
            all_docs = valid_docs

    # Final safety clamp to k_total (in case disjoint sets exceeded total)
    if max_docs and len(all_docs) > max_docs:
        logger.info(f"Clamping final merged documents from {len(all_docs)} to {max_docs}")
        all_docs = all_docs[:max_docs]

    # 6. Format sources for response
    retrieval = format_sources(all_docs)
    context = retrieval["context"]

    # 7. Generate Response
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

    # 7.5 POST-VALIDATION: Check for hallucinated developer attributions
    is_grounded, grounding_issues = validate_response_grounding(
        response=final_response,
        retrieved_docs=all_docs,
        claimed_entities=metadata_filters
    )
    if not is_grounded:
        logger.warning(f"Grounding issues detected: {grounding_issues}")
        # Add warning to output if not already warned pre-query
        if not entity_warning:
            entity_warning = "**⚠️ Grounding Warning:** " + "; ".join(grounding_issues) + "\n\n"

    logger.info("=" * 50)
    logger.success("THINKING COMPLETE")
    logger.info("=" * 50)

    # 8. Build structured output
    question_type_label = question_type.value.replace("_", " ").title()
    validation_status = "Fixed ⚠" if was_fixed else "Passed ✓"
    grounding_status = "Grounded ✓" if is_grounded else "Warning ⚠"

    if lang == 'spanish':
        thought_summary = f"""**💭 Proceso de análisis:**
- Tipo de pregunta: {question_type_label}
- Consultas generadas: {len(queries)}
- Documentos recuperados: {len(all_docs)}
- Validación: {"Corregida ⚠" if was_fixed else "Pasada ✓"}
- Fundamentación: {"Verificada ✓" if is_grounded else "Advertencia ⚠"}

---

**📋 Respuesta:**
"""
    else:
        thought_summary = f"""**💭 Analysis process:**
- Question type: {question_type_label}
- Queries generated: {len(queries)}
- Documents retrieved: {len(all_docs)}
- Validation: {validation_status}
- Grounding: {grounding_status}

---

**📋 Answer:**
"""

    # Stream output
    yield thought_summary

    # If there's an entity warning, show it before the response
    if entity_warning:
        yield entity_warning

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

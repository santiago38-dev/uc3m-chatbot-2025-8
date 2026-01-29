"""
Vector Store - Loads ChromaDB from Person B pipeline.

Enhanced with:
- Multi-value filter extraction for comparative queries
- Alias expansion for parent companies and TSPs
- Hard filtering support with $in operator
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Import filter utilities (separated to avoid circular imports)
from src.rag_advanced.filter_utils import (
    extract_multi_filters_from_query,
    build_chromadb_where_clause,
    PARENT_PATTERNS,
    TSP_PATTERNS
)

load_dotenv()

# Load from .env or use default relative path
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./output/chromadb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sgia_chunks")


def get_vectorstore(chromadb_path: str = None) -> Chroma:
    """Load the ChromaDB vector store."""
    path = chromadb_path or CHROMADB_PATH

    if not Path(path).exists():
        raise FileNotFoundError(f"ChromaDB not found at: {path}")

    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=path,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )

    return vectorstore


def get_retriever(k_docs: int = 10, filters: dict = None, chromadb_path: str = None):
    """Get a standard LangChain retriever."""
    vectorstore = get_vectorstore(chromadb_path)
    search_kwargs = {"k": k_docs}

    if filters:
        if len(filters) == 1:
            key, val = list(filters.items())[0]
            search_kwargs["filter"] = {key: {"$eq": val}}
        else:
            search_kwargs["filter"] = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}

    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def extract_filters_from_query(query: str) -> dict:
    """
    Extract metadata filters from query text using regex patterns.

    Returns a dict with single values for backward compatibility.
    Use extract_multi_filters_from_query() for comparative queries.
    """
    filters = {}
    q = query.lower()

    def has_match(pattern: str) -> bool:
        return bool(re.search(rf"\b({pattern})\b", q, re.IGNORECASE))

    # Zone
    if has_match(r"coast|coastal|houston"):
        filters['zone'] = 'COAST'
    elif has_match(r"west|western"):
        filters['zone'] = 'WEST'
    elif has_match(r"north|northern"):
        filters['zone'] = 'NORTH'
    elif has_match(r"south|southern"):
        filters['zone'] = 'SOUTH'
    elif has_match(r"panhandle"):
        filters['zone'] = 'PANHANDLE'

    # Fuel Type
    if has_match(r"battery|bateria|baterias|storage|bess"):
        filters['fuel_type'] = 'OTH'
    elif has_match(r"solar|solares|pv|sun"):
        filters['fuel_type'] = 'SOL'
    elif has_match(r"wind|viento|vientos"):
        filters['fuel_type'] = 'WIN'
    elif has_match(r"gas|natural gas"):
        filters['fuel_type'] = 'GAS'

    # Technology
    if has_match(r"pv|solar|solares"):
        filters['technology'] = 'PV'
    elif has_match(r"gas turbine|gt|turbina"):
        filters['technology'] = 'GT'

    # TSP (Transmission Service Provider)
    if has_match(r"centerpoint|cnp|cpnt"):
        filters['tsp_normalized'] = 'CENTERPOINT'
    elif has_match(r"oncor"):
        filters['tsp_normalized'] = 'ONCOR'

    # Parent Company
    if has_match(r"nextera"):
        filters['parent_company'] = 'NEXTERA'
    elif has_match(r"rwe"):
        filters['parent_company'] = 'RWE'
    elif has_match(r"engie"):
        filters['parent_company'] = 'ENGIE'

    # County
    if has_match(r"brazoria"):
        filters['county'] = 'Brazoria'
    elif has_match(r"harris"):
        filters['county'] = 'Harris'
    elif has_match(r"matagorda"):
        filters['county'] = 'Matagorda'
    elif has_match(r"bell"):
        filters['county'] = 'Bell'

    return filters


# =============================================================================
# ENHANCED MULTI-VALUE FILTER EXTRACTION
# Functions imported from filter_utils.py to avoid circular imports
# Re-exported here for backward compatibility
# =============================================================================

# extract_multi_filters_from_query and build_chromadb_where_clause
# are imported at top of file from src.rag_advanced.filter_utils


def similarity_search_with_boost(
    vectorstore,
    query: str,
    k: int = 10,
    boost_factor: float = 0.8,
    k_initial: int = 50,
    external_filters: dict = None
) -> list:
    """
    Performs a soft-filtered search. Instead of excluding docs, it boosts
    those that match the query's metadata filters.
    """
    filters = extract_filters_from_query(query)

    # Merge external filters (e.g. from LLM extraction)
    if external_filters:
        filters.update(external_filters)

    # 1. Wide search
    results = vectorstore.similarity_search_with_score(query, k=k_initial)

    # 2. Apply boosting
    boosted_results = []
    for doc, score in results:
        original_score = score
        match_count = 0

        # Check metadata matches
        for key, val in filters.items():
            if doc.metadata.get(key) == val:
                match_count += 1
                # SPECIAL CASE: Project Name match gets massive boost
                if key == 'project_name':
                     match_count += 10 # Artificially inflate match count to prioritize project matches above all else

        # Apply boost if there's a match (lower distance is better)
        if match_count > 0:
            effective_boost = boost_factor ** match_count
            score = score * effective_boost

        boosted_results.append((doc, score, original_score, match_count))

    # 3. Re-sort by boosted score
    boosted_results.sort(key=lambda x: x[1])

    return boosted_results[:k]


class SmartRetriever:
    """
    A retriever that uses boosted similarity search based on query metadata.
    Compatible with LCEL through the invoke() method.

    Enhanced with:
    - Hard filtering mode for comparative queries (using $in operator with alias expansion)
    - Soft filtering (boosting) for general queries
    """

    def __init__(
        self,
        vectorstore,
        k: int = 20,  # Increased from 15 for better coverage
        boost_factor: float = 0.8,
        k_initial: int = 50
    ):
        self.vectorstore = vectorstore
        self.k = k
        self.boost_factor = boost_factor
        self.k_initial = k_initial

    def invoke(self, query: str) -> list:
        """LCEL-compatible invoke method."""
        return self._search(query)

    def __call__(self, query: str) -> list:
        """Allow direct calling."""
        return self._search(query)

    def search_with_filters(self, query: str, filters: dict = None) -> list:
        """Explicitly search with external filters (soft boosting)."""
        return self._search(query, external_filters=filters)

    def search_with_hard_filters(
        self,
        query: str,
        where: dict = None,
        k: int = None
    ) -> list:
        """
        Search with HARD ChromaDB filters (no boosting, direct $in/$eq).

        This is critical for comparative queries where we MUST retrieve
        documents from specific developers/TSPs.

        Args:
            query: Search query
            where: ChromaDB where clause (from build_chromadb_where_clause)
            k: Number of results (defaults to self.k)

        Returns:
            List of documents
        """
        effective_k = k or self.k

        if where:
            # Hard filter: use ChromaDB's where clause directly
            try:
                results = self.vectorstore.similarity_search(
                    query,
                    k=effective_k,
                    filter=where
                )
                return results
            except Exception as e:
                # Fallback to no filter if ChromaDB rejects the clause
                print(f"Warning: Hard filter failed ({e}), falling back to unfiltered search")
                return self.vectorstore.similarity_search(query, k=effective_k)
        else:
            return self.vectorstore.similarity_search(query, k=effective_k)

    def search_comparative(self, query: str, k: int = None) -> Tuple[list, Dict[str, Any]]:
        """
        Enhanced search for comparative queries.

        Automatically extracts multi-value filters, expands aliases,
        and applies hard filtering.

        Args:
            query: The user's question
            k: Number of results

        Returns:
            Tuple of (documents, filter_info)
            filter_info contains extracted filters for debugging/display
        """
        effective_k = k or self.k

        # Extract multi-value filters
        filters = extract_multi_filters_from_query(query)

        # Build ChromaDB where clause with alias expansion
        where_clause = build_chromadb_where_clause(filters, expand_aliases=True)

        # Perform hard-filtered search
        docs = self.search_with_hard_filters(query, where=where_clause, k=effective_k)

        # Return docs and filter info for transparency
        filter_info = {
            'extracted_filters': filters,
            'where_clause': where_clause,
            'is_comparative': isinstance(filters.get('parent_company'), list) or
                            isinstance(filters.get('tsp_normalized'), list) or
                            isinstance(filters.get('fuel_type'), list)
        }

        return docs, filter_info

    def _search(self, query: str, external_filters: dict = None) -> list:
        """Perform boosted similarity search (soft filtering)."""
        boosted_results = similarity_search_with_boost(
            vectorstore=self.vectorstore,
            query=query,
            k=self.k,
            boost_factor=self.boost_factor,
            k_initial=self.k_initial,
            external_filters=external_filters
        )
        # Return only the documents (not scores)
        return [doc for doc, score, orig_score, match_count in boosted_results]

    def get_relevant_documents(self, query: str) -> list:
        """LangChain retriever interface compatibility."""
        return self._search(query)

    def get_all_projects_by_threshold(
        self,
        threshold_field: str,
        threshold_value: float,
        operator: str = "$gte"
    ) -> List[Dict[str, Any]]:
        """
        Get ALL unique projects meeting a threshold criteria.

        This bypasses semantic search to directly query metadata for
        threshold queries like "projects with >$100/kW".

        Args:
            threshold_field: Metadata field to filter (e.g., 'security_per_kw')
            threshold_value: Numeric threshold value
            operator: Comparison operator ('$gte', '$gt', '$lte', '$lt')

        Returns:
            List of unique project dicts with key metadata
        """
        try:
            # Access underlying Chroma collection
            collection = self.vectorstore._collection

            # Query with numeric filter - get more results to ensure we find all projects
            where_clause = {threshold_field: {operator: threshold_value}}

            results = collection.get(
                where=where_clause,
                include=["metadatas"]
            )

            # Extract unique projects with their metadata
            seen_projects = set()
            unique_projects = []

            for metadata in results.get('metadatas', []):
                project_name = metadata.get('project_name', '')
                inr = metadata.get('inr', '')
                key = f"{project_name}|{inr}"

                if key not in seen_projects and project_name:
                    seen_projects.add(key)
                    unique_projects.append({
                        'project_name': project_name,
                        'inr': inr,
                        'developer': metadata.get('developer_spv', metadata.get('parent_company', 'N/A')),
                        'fuel_type': metadata.get('fuel_type', 'N/A'),
                        'zone': metadata.get('zone', 'N/A'),
                        'capacity_mw': metadata.get('capacity_mw', 'N/A'),
                        'security_amount': metadata.get('security_amount', 'N/A'),
                        'security_per_kw': metadata.get('security_per_kw', 'N/A'),
                        'tsp': metadata.get('tsp_normalized', metadata.get('tsp', 'N/A'))
                    })

            # Sort by security_per_kw descending
            unique_projects.sort(
                key=lambda x: x.get('security_per_kw', 0) if x.get('security_per_kw') != 'N/A' else 0,
                reverse=True
            )

            return unique_projects

        except Exception as e:
            print(f"Warning: Threshold query failed ({e})")
            return []


def get_smart_retriever(
    k_docs: int = 20,  # Increased from 15 for better coverage
    chromadb_path: str = None,
    boost_factor: float = 0.8,
    k_initial: int = 50
) -> SmartRetriever:
    """
    Creates a SmartRetriever that uses boosted similarity search.

    The retriever automatically detects metadata filters from the query
    and boosts matching documents instead of hard-filtering.

    Args:
        k_docs: Number of documents to return
        chromadb_path: Path to ChromaDB
        boost_factor: Multiplier for matching docs (lower = more boost, since distance is minimized)
        k_initial: Initial pool size for re-ranking

    Returns:
        SmartRetriever instance compatible with LCEL
    """
    vectorstore = get_vectorstore(chromadb_path)
    return SmartRetriever(
        vectorstore=vectorstore,
        k=k_docs,
        boost_factor=boost_factor,
        k_initial=k_initial
    )


def get_hybrid_retriever(
    k_docs: int = 20,  # Increased from 15 for better coverage (matches SmartRetriever default)
    chromadb_path: str = None,
    use_smart: bool = True,
    boost_factor: float = 0.8
):
    """
    Factory function to get either smart (boosted) or standard retriever.

    Args:
        k_docs: Number of documents to return
        chromadb_path: Path to ChromaDB
        use_smart: If True, uses SmartRetriever with boosting
        boost_factor: Boost factor for smart retriever

    Returns:
        Retriever instance
    """
    if use_smart:
        return get_smart_retriever(k_docs=k_docs, chromadb_path=chromadb_path, boost_factor=boost_factor)
    else:
        return get_retriever(k_docs=k_docs, chromadb_path=chromadb_path)


def get_document_content(project_name: str, inr: str, section: str) -> str:
    """
    Retrieve full content for a specific document section using metadata.
    """
    vectorstore = get_vectorstore()

    # Use Chroma's internal get method for metadata filtering
    # Note: langchain wrapper might expose .get but _collection is direct access to underlying chromadb
    try:
        where_clause = {
            "$and": [
                {"project_name": {"$eq": project_name}},
                {"inr": {"$eq": inr}},
                {"section": {"$eq": section}}  # Use section (article_1, exhibit_c, etc.)
            ]
        }

        # We need to access the underlying collection to use 'where' efficiently without embeddings
        results = vectorstore._collection.get(where=where_clause)
        documents = results.get("documents", [])

        if not documents:
            return f"No content found for {project_name} ({inr}) - {section}"

        # Concatenate all chunks found for this section (usually it's one or a few)
        return "\n\n---\n\n".join(documents)

    except Exception as e:
        return f"Error retrieving document: {str(e)}"


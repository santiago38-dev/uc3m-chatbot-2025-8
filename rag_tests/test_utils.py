"""
RAG Chain Quality Evaluation

Tests the RAG chain used in frontend.py using multiple metrics:
- Recall@K: Retrieval quality metric
- Mean Reciprocal Rank (MRR): Retrieval ranking quality
- FactScore: Factual accuracy following Min et al. (2023) methodology
  (FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation)
  Paper: https://aclanthology.org/2023.emnlp-main.741/
- BERTScore: Semantic similarity between generated and reference answers
"""

import time
import statistics
import re
from typing import List, Dict, Any

from src.vector_store import get_retriever, get_document_content
from src.rag_advanced.chain import get_rag_chain, OOS_QUESTION_MSG
from src.rag_advanced.utils import RAGMode, detect_language
from src.llm_client import call_llm_api_full

try:
    from bert_score import score as bert_score_fn
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: bert_score not available. BERTScore will be skipped.")


class RAGQualityEvaluator:
    """Evaluates RAG chain quality using multiple metrics."""

    def __init__(
        self,
        k_docs: int = 10,
        mode: RAGMode = RAGMode.FLASH,
        with_summary: bool = False
    ):
        """
        Initialize evaluator.

        Args:
            k_docs: Number of documents to retrieve
            mode: RAG mode (FLASH or THINKING)
            with_summary: Whether to include summary in responses
        """
        self.k_docs = k_docs
        self.mode = mode
        self.with_summary = with_summary
        self.retriever = get_retriever(k_docs=k_docs)
        self.rag_chain = get_rag_chain(
            retriever=self.retriever,
            mode=mode,
            k_docs=k_docs,
            with_history=True,
            with_summary=with_summary
        )

    def _document_to_key(self, doc) -> str:
        """Convert document to a unique key for comparison."""
        meta = doc.metadata
        project = meta.get('project_name', 'Unknown')
        inr = meta.get('inr', 'N/A')
        # Use section_type first (primary field), fall back to section
        section = meta.get('section_type') or meta.get('section', 'N/A')
        return f"{project}::{inr}::{section}"

    @staticmethod
    def doc_to_key(project_name: str, inr: str, section: str) -> str:
        """Static helper to create document key from metadata."""
        return f"{project_name}::{inr}::{section}"

    def compute_recall_at_k(
        self,
        retrieved_doc_keys: List[str],
        relevant_doc_keys: List[str],
        k: int
    ) -> float:
        """
        Compute Recall@K for retrieval quality.

        Args:
            retrieved_doc_keys: List of keys for retrieved documents
            relevant_doc_keys: List of keys for relevant documents
            k: Number of top documents to consider

        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not relevant_doc_keys:
            return 0.0

        # Get top-k retrieved documents
        top_k_docs = retrieved_doc_keys[:k]
        retrieved_keys = set(top_k_docs)
        relevant_keys = set(relevant_doc_keys)

        # Recall = |retrieved ∩ relevant| / |relevant|
        intersection = retrieved_keys & relevant_keys
        return len(intersection) / len(relevant_keys) if relevant_keys else 0.0

    def compute_mrr(
        self,
        retrieved_keys: List[str],
        relevant_doc_keys: List[str]
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).

        Args:
            retrieved_keys: List of keys for retrieved documents (ordered by relevance)
            relevant_doc_keys: List of keys for relevant documents

        Returns:
            Reciprocal rank of first relevant document (0.0 to 1.0)
        """
        if not relevant_doc_keys:
            return 0.0

        relevant_keys = set(relevant_doc_keys)

        # Find rank of first relevant document (1-indexed)
        for rank, doc_key in enumerate(retrieved_keys, start=1):
            if doc_key in relevant_keys:
                return 1.0 / rank

        # No relevant document found
        return 0.0

    def _extract_atomic_facts(self, text: str) -> List[str]:
        """
        Extract atomic facts from text following FactScore methodology.

        Atomic facts are simple, verifiable statements. This method uses
        LLM to decompose the text into atomic facts.

        Args:
            text: Input text to extract facts from

        Returns:
            List of atomic fact strings
        """
        prompt = f"""Break down the following text into a list of atomic facts.
An atomic fact is a simple, specific, verifiable claim that can be independently checked.

Text:
{text}

Extract all atomic facts. Each fact should be:
- Specific and concrete (not vague)
- Independently verifiable
- A single statement (not multiple claims)

Return the facts as a numbered list, one per line.
If there are no facts, return "NONE".

Examples of atomic facts:
- "The security deposit is specified in Exhibit A."
- "Article 5 specifies milestones including commercial operation date."
- "The project capacity is 100 MW."

Facts:"""

        try:
            result = call_llm_api_full(prompt).strip()
            facts = []
            for line in result.split('\n'):
                line = line.strip()
                # Remove numbering (e.g., "1. ", "- ", etc.)
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                line = re.sub(r'^[-*]\s*', '', line)
                if line and line.upper() != "NONE" and len(line) > 10:
                    facts.append(line)
            return facts
        except Exception as e:
            print(f"    Warning: Atomic fact extraction failed: {e}")
            return []

    def _verify_fact_against_context(self, fact: str, context: str) -> bool:
        """
        Verify if a single atomic fact is supported by the context.

        Args:
            fact: Atomic fact to verify
            context: Context/knowledge source to check against

        Returns:
            True if fact is supported, False otherwise
        """
        prompt = f"""You are verifying if a factual claim is supported by the provided context.

Atomic Fact to verify:
{fact}

Context (knowledge source):
{context[:4000]}

Determine if the atomic fact is supported by the context. A fact is supported if:
1. The context contains information that directly or strongly implies the fact
2. The fact is consistent with the information in the context
3. There is no contradictory information in the context

Respond with ONLY "YES" if the fact is supported, or "NO" if it is not supported or cannot be verified.

Response:"""

        try:
            result = call_llm_api_full(prompt).strip().upper()
            return "YES" in result
        except Exception as e:
            print(f"      Warning: Fact verification failed: {e}")
            return False

    def compute_factscore(
        self,
        generated_answer: str,
        context: str
    ) -> float:
        """
        Compute FactScore following the methodology from:
        Min et al. (2023). FActScore: Fine-grained Atomic Evaluation of
        Factual Precision in Long Form Text Generation. EMNLP 2023.
        https://aclanthology.org/2023.emnlp-main.741/

        FactScore breaks a generation into atomic facts and computes the
        percentage of atomic facts supported by a reliable knowledge source.
        In RAG evaluation, we use the retrieved context as the knowledge source.

        Args:
            generated_answer: The generated answer from RAG chain
            context: Retrieved context used as knowledge source for verification

        Returns:
            FactScore (0.0 to 1.0) - percentage of atomic facts supported by context
        """
        if not context or not context.strip():
            return 0.0

        # Extract main answer (remove sources section)
        answer = generated_answer.split("Sources:")[0].split("Fuentes:")[0].strip()

        if not answer or len(answer) < 10:
            return 0.0

        try:
            # Step 1: Extract atomic facts from generated answer
            atomic_facts = self._extract_atomic_facts(answer)

            if not atomic_facts:
                # If we can't extract facts, fall back to simpler verification
                # Check if the overall answer is supported
                return 1.0 if self._verify_fact_against_context(answer, context) else 0.5

            # Step 2: Verify each atomic fact against the context
            supported_count = 0
            for fact in atomic_facts:
                if self._verify_fact_against_context(fact, context):
                    supported_count += 1

            # Step 3: Compute FactScore as percentage of supported facts
            factscore = supported_count / len(atomic_facts) if atomic_facts else 0.0
            return factscore

        except Exception as e:
            print(f"  Warning: FactScore computation failed: {e}")
            # Fallback: simple verification of the whole answer
            try:
                is_supported = self._verify_fact_against_context(answer, context)
                return 1.0 if is_supported else 0.5
            except:
                return 0.5

    def compute_bertscore(
        self,
        generated_answer: str,
        reference_answer: str
    ) -> float:
        """
        Compute BERTScore - semantic similarity metric.

        Args:
            generated_answer: Generated answer
            reference_answer: Reference answer

        Returns:
            BERTScore F1 (0.0 to 1.0)
        """
        if not BERT_AVAILABLE:
            return 0.0

        if not reference_answer or not generated_answer:
            return 0.0

        # Extract main answer (remove sources section)
        gen_clean = generated_answer.split("Sources:")[0].split("Fuentes:")[0].strip()
        ref_clean = reference_answer.strip()

        try:
            P, R, F1 = bert_score_fn(
                [gen_clean],
                [ref_clean],
                model_type="distilbert-base-uncased",
                verbose=False
            )
            return F1.mean().item()
        except Exception as e:
            print(f"  Warning: BERTScore computation failed: {e}")
            return 0.0

    def check_out_of_scope(self, response: str, is_in_scope: bool):
        test = response in OOS_QUESTION_MSG.values()
        return test == (not is_in_scope)

    def _parse_rag_response(self, generated_answer: str):
        sources_split = generated_answer.split("Sources:\n", 1)
        main_response = sources_split[0].strip()
        sources_content = sources_split[1].strip() if len(sources_split) > 1 else None
        sources = {"keys": [], "coords": []}

        if not sources_content:
            return main_response, sources

        lines = sources_content.strip().split('\n')
        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                continue
            # Parse metadata: [1] Project (INR) - Section
            # Regex: find optional [N], then Project, (INR), -, Section
            match = re.search(r'(?:\[\d+\]\s*)?(.*?)\s*\((.*?)\)\s*-\s*(.*)', clean_line)
            if not match:
                continue

            project = match.group(1).strip()
            inr = match.group(2).strip()
            section = match.group(3).strip()
            sources["keys"].append(RAGQualityEvaluator.doc_to_key(project, inr, section))
            sources["coords"].append({"project": project, "inr": inr, "section": section})

        return main_response, sources


    def evaluate(
        self,
        dataset: List[Dict[str, Any]],
        k_values: List[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG chain on dataset.

        Expected dataset format:
        [
            {
                "question": str,
                "lang": str,
                "is_in_scope": bool,
                "reference_answer": str,
                "relevant_doc_keys": List[str],  # e.g., ["Project::INR::Section", ...]
            },
            ...
        ]

        Args:
            dataset: List of test cases
            k_values: List of K values for Recall@K (default: [1, 5, 10])

        Returns:
            Dictionary with evaluation results
        """
        if k_values is None:
            k_values = [1, 5, min(10, self.k_docs)]

        print(f"\n{'='*80}")
        print(f"RAG Chain Quality Evaluation")
        print(f"{'='*80}")
        print(f"Mode: {self.mode.value}")
        print(f"K docs: {self.k_docs}")
        print(f"Evaluating {len(dataset)} test cases...\n")

        metrics = {
            "lang_accuracy": [],
            "reject_accuracy": [],
            "recall_at_k": {k: [] for k in k_values},
            "mrr": [],
            "factscore": [],
            "bertscore": [],
            "latency": [],
        }

        for i, case in enumerate(dataset, 1):
            question = case["question"]
            reference_answer = case.get("reference_answer", "")
            relevant_doc_keys = case.get("relevant_doc_keys", [])

            print(f"[{i}/{len(dataset)}] {question[:120]}...")

            # Get generated answer
            start_time = time.time()
            try:
                config = {"configurable": {"session_id": f"eval_{i}"}}
                # Collect streamed response (chain returns generator)
                generated_answer = ""
                for chunk in self.rag_chain.stream(
                    {"question": question},
                    config=config
                ):
                    generated_answer += str(chunk)
                # Clean up the answer (remove extra whitespace from streaming)
                generated_answer = generated_answer.strip()
            except Exception as e:
                print(f"  ERROR in generation: {e}")
                generated_answer = "ERROR"

            latency = time.time() - start_time
            metrics["latency"].append(latency)

            # Split in main response and sources
            main_response, docs = self._parse_rag_response(generated_answer)

            # Check language and out-of-scope rejection
            generated_lang = detect_language(main_response)
            metrics["lang_accuracy"].append(generated_lang == case["lang"])

            reject_hit = self.check_out_of_scope(main_response, case["is_in_scope"])
            metrics["reject_accuracy"].append(reject_hit)
            print(f"  Language hit: {metrics['lang_accuracy'][-1]}, "
                  f"Reject/accept hit: {metrics['reject_accuracy'][-1]}")

            # Skip further tests if question is actually out-of-scope
            if not case["is_in_scope"]:
                continue

            # Compute retrieval metrics (Recall@K, MRR)
            if relevant_doc_keys:
                for k in k_values:
                    recall = self.compute_recall_at_k(docs["keys"], relevant_doc_keys, k)
                    metrics["recall_at_k"][k].append(recall)

                mrr = self.compute_mrr(docs["keys"], relevant_doc_keys)
                metrics["mrr"].append(mrr)
                print(f"  Recall@1: {metrics['recall_at_k'][1][-1]:.2f}, "
                      f"Recall@5: {metrics['recall_at_k'][5][-1]:.2f}, "
                      f"MRR: {mrr:.2f}")
            else:
                print("  No relevant doc keys provided, skipping retrieval metrics")

            # Compute generation metrics (FactScore, BERTScore)
            if reference_answer and generated_answer != "ERROR":
                # Get context for FactScore
                db_docs = [get_document_content(d["project"], d["inr"], d["section"]) for d in docs["coords"][:5]]
                context = "\n".join(db_docs)

                factscore = self.compute_factscore(
                    generated_answer,
                    context
                )
                metrics["factscore"].append(factscore)

                bertscore = self.compute_bertscore(
                    generated_answer,
                    reference_answer
                )
                metrics["bertscore"].append(bertscore)

                print(f"  FactScore: {factscore:.2f}, BERTScore: {bertscore:.2f}, "
                      f"Latency: {latency:.2f}s")
            else:
                print(f"  No reference answer, skipping generation metrics. "
                      f"Latency: {latency:.2f}s")

            print()

        # Compute aggregated results
        results = self._aggregate_results(metrics, k_values)
        self._print_report(results, k_values)

        return results

    def _aggregate_results(
        self,
        metrics: Dict[str, Any],
        k_values: List[int]
    ) -> Dict[str, Any]:
        """Aggregate metrics across all test cases."""
        def avg(lst):
            return statistics.mean(lst) if lst else 0.0

        results = {
            "recall_at_k": {
                k: avg(metrics["recall_at_k"][k])
                for k in k_values
            },
            "lang_accuracy": avg(metrics["lang_accuracy"]),
            "reject_accuracy": avg(metrics["reject_accuracy"]),
            "mrr": avg(metrics["mrr"]),
            "factscore": avg(metrics["factscore"]),
            "bertscore": avg(metrics["bertscore"]),
            "latency": avg(metrics["latency"]),
            "num_cases": len(metrics["latency"])
        }

        return results

    def _print_report(self, results: Dict[str, Any], k_values: List[int]):
        """Print evaluation report."""
        print("\n" + "="*80)
        print("EVALUATION REPORT")
        print("="*80)
        print(f"{'Metric':<25} {'Value':<15} {'Description'}")
        print("-"*80)


        print(f"{'Language accuracy':<25} {results['lang_accuracy']:>6.2%}     "
              f"Accuracy for language detection")
        print(f"{'Reject accuracy':<25} {results['reject_accuracy']:>6.2%}     "
              f"Accuracy for out-of-scope questions detection")

        # Retrieval metrics
        for k in k_values:
            print(f"{'Recall@' + str(k):<25} {results['recall_at_k'][k]:>6.2%}     "
                  f"Fraction of relevant docs in top {k}")

        print(f"{'Mean Reciprocal Rank (MRR)':<25} {results['mrr']:>6.2%}     "
              f"Average reciprocal rank of first relevant doc")

        # Generation metrics
        print(f"{'FactScore':<25} {results['factscore']:>6.2f}     "
              f"Factual accuracy (0.0-1.0)")
        print(f"{'BERTScore':<25} {results['bertscore']:>6.2f}     "
              f"Semantic similarity (0.0-1.0)")
        print(f"{'Avg Latency':<25} {results['latency']:>6.2f}s    "
              f"Average response time")
        print("-"*80)
        print(f"Total test cases: {results['num_cases']}")
        print("="*80 + "\n")


# Sample test dataset
# Note: You should replace this with your actual test dataset
# with questions, reference answers, and relevant document keys
DEFAULT_DATASET = [
    {
        "question": "What are the security deposit requirements for interconnection?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "The Generator shall provide security as specified in Exhibit A. The security amount depends on the project capacity and type.",
        "relevant_doc_keys": []  # Add actual document keys like ["ProjectName::INR::Article 11"]
    },
    {
        "question": "What are the milestone requirements in Article 5?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "Article 5 specifies milestones including commercial operation date and construction deadlines.",
        "relevant_doc_keys": []  # Add actual document keys
    },
]


def run_evaluation(
    dataset: List[Dict[str, Any]] = None,
    k_docs: int = 10,
    mode: RAGMode = RAGMode.FLASH,
    with_summary: bool = False,
    k_values: List[int] = None
) -> Dict[str, Any]:
    """
    Run RAG quality evaluation.

    Args:
        dataset: Test dataset (uses DEFAULT_DATASET if None)
            Each entry should have:
            - "question": str
            - "lang": str ("english" or "spanish")
            - "is_in_scope": bool
            - "reference_answer": str (optional for retrieval-only metrics)
            - "relevant_doc_keys": List[str] (optional for generation-only metrics)
              Format: ["ProjectName::INR::Section", ...]
              Use RAGQualityEvaluator.doc_to_key() helper to create keys
        k_docs: Number of documents to retrieve
        mode: RAG mode (FLASH or THINKING)
        with_summary: Whether to include summary in responses
        k_values: List of K values for Recall@K (default: [1, 5, min(10, k_docs)])

    Returns:
        Dictionary with evaluation results

    Example:
        dataset = [
            {
                "question": "What is the security deposit?",
                "lang": "english",
                "is_in_scope": True,
                "reference_answer": "The security deposit is specified in Exhibit A.",
                "relevant_doc_keys": [
                    RAGQualityEvaluator.doc_to_key("ProjectName", "25INR0494", "Article 11")
                ]
            }
        ]
        results = run_evaluation(dataset, mode=RAGMode.FLASH)
    """
    if dataset is None:
        dataset = DEFAULT_DATASET
        print("Warning: Using sample dataset. Replace with your actual test data.")

    evaluator = RAGQualityEvaluator(
        k_docs=k_docs,
        mode=mode,
        with_summary=with_summary
    )

    results = evaluator.evaluate(dataset, k_values=k_values)
    return results

# =============================================================================
# DATASET VALIDATION HELPERS
# =============================================================================

def validate_dataset(dataset):
    """Validate that all required fields are present and correct."""
    required_fields = ["question", "lang", "is_in_scope", "reference_answer", "relevant_doc_keys"]
    valid_langs = ["english", "spanish"]

    errors = []

    for i, case in enumerate(dataset):
        # Check required fields
        for field in required_fields:
            if field not in case:
                errors.append(f"Case {i}: Missing field '{field}'")

        # Check lang field
        if case.get("lang") not in valid_langs:
            errors.append(f"Case {i}: Invalid lang '{case.get('lang')}'")

        # Check is_in_scope type
        if not isinstance(case.get("is_in_scope"), bool):
            errors.append(f"Case {i}: is_in_scope must be boolean")

        # Check relevant_doc_keys type
        if not isinstance(case.get("relevant_doc_keys"), list):
            errors.append(f"Case {i}: relevant_doc_keys must be a list")

        # For in-scope questions, check for non-empty reference and doc_keys
        if case.get("is_in_scope"):
            if not case.get("reference_answer"):
                errors.append(f"Case {i}: In-scope question missing reference_answer")
            if not case.get("relevant_doc_keys"):
                errors.append(f"Case {i}: In-scope question missing relevant_doc_keys")

    if errors:
        print("VALIDATION ERRORS:")
        for error in errors:
            print(f" {error}")
        return False
    else:
        print(" Dataset validation passed!")
        return True


def print_coverage_stats(dataset):
    """Print dataset coverage statistics."""
    stats = {
        "total": len(dataset),
        "out_of_scope_en": 0,
        "out_of_scope_es": 0,
        "in_scope_en": 0,
        "in_scope_es": 0,
        "fuel_types": set(),
        "zones": set(),
    }

    for case in dataset:
        is_in_scope = case["is_in_scope"]
        lang = case["lang"]

        if not is_in_scope:
            if lang == "english":
                stats["out_of_scope_en"] += 1
            else:
                stats["out_of_scope_es"] += 1
        else:
            if lang == "english":
                stats["in_scope_en"] += 1
            else:
                stats["in_scope_es"] += 1

    print("\n" + "="*60)
    print("DATASET COVERAGE STATISTICS")
    print("="*60)
    print(f"Total test cases: {stats['total']}")
    print(f"\nOut-of-scope questions:")
    print(f"  English: {stats['out_of_scope_en']}")
    print(f"  Spanish: {stats['out_of_scope_es']}")
    print(f"\nIn-scope questions:")
    print(f"  English: {stats['in_scope_en']}")
    print(f"  Spanish: {stats['in_scope_es']}")
    print("="*60)


if __name__ == "__main__":
    # Example usage
    import sys

    mode_str = sys.argv[1] if len(sys.argv) > 1 else "flash"
    mode = RAGMode.FLASH if mode_str.lower() == "flash" else RAGMode.THINKING

    print("="*80)
    print("RAG Chain Quality Evaluation")
    print("="*80)
    print(f"\nTo use this evaluator:")
    print("1. Prepare a dataset with 'question', 'reference_answer', and 'relevant_doc_keys'")
    print("2. Document keys should be in format: 'ProjectName::INR::Section'")
    print("3. Call run_evaluation() with your dataset\n")
    print("Example dataset entry:")
    print("  from test_rag_quality import RAGQualityEvaluator")
    print("  {")
    print("    'question': 'What is the security deposit?',")
    print("    'lang': 'english',")
    print("    'is_in_scope': True,")
    print("    'reference_answer': 'The security deposit is...',")
    print("    'relevant_doc_keys': [")
    print("      RAGQualityEvaluator.doc_to_key('ProjectName', '25INR0494', 'Article 11')")
    print("    ]")
    print("  }\n")

    results = run_evaluation(mode=mode)

    print("\nEvaluation complete!")


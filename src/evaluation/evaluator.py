import re
from typing import List, Dict, Any, Optional

# pyrefly: ignore [missing-import]
from src.config import settings
# pyrefly: ignore [missing-import]
from src.retrieval.retriever import HybridRetriever
# pyrefly: ignore [missing-import]
from src.generation.generator import GroundedGenerator

class RAGEvaluator:
    """Evaluates RAG pipeline outputs using LLM-as-judge and rule-based metrics."""
    def __init__(self, retriever: Optional[HybridRetriever] = None, generator: Optional[GroundedGenerator] = None):
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or GroundedGenerator()
        self.openai_client = self.generator.openai_client

    def evaluate_correctness(self, answer: str, ground_truth: str) -> float:
        """LLM-as-judge score for semantic correctness between generated answer and ground truth."""
        system_prompt = (
            "You are an expert quality assurance judge. Your task is to evaluate the correctness of a generated "
            "answer compared to a reference ground truth answer.\n\n"
            "Instructions:\n"
            "1. Output a score between 0.0 and 1.0 (inclusive).\n"
            "2. 1.0 means the generated answer is semantically complete and fully correct compared to the ground truth.\n"
            "3. 0.0 means the generated answer is completely incorrect, irrelevant, or contradicts the ground truth.\n"
            "4. Note: If the ground truth indicates no info is available (unanswerable) and the generated answer "
            "correctly states the fallback info, it is 1.0.\n"
            "5. Output ONLY the numeric float score (e.g., 1.0, 0.8, 0.5, 0.0). Do NOT include any other text."
        )
        
        user_prompt = (
            f"Ground Truth Reference Answer:\n"
            f"\"\"\"\n"
            f"{ground_truth.strip()}\n"
            f"\"\"\"\n\n"
            f"Generated Answer to Evaluate:\n"
            f"\"\"\"\n"
            f"{answer.strip()}\n"
            f"\"\"\"\n"
        )
        
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=5
            )
            score_str = (response.choices[0].message.content or "").strip()
            match = re.search(r"\d+(?:\.\d+)?", score_str)
            if match:
                score = float(match.group(0))
                return min(max(score, 0.0), 1.0)
            return 0.0
        except Exception as e:
            print(f"Error during correctness check: {e}")
            return 0.0

    def evaluate_faithfulness(self, answer: str, retrieved_context: str) -> float:
        """LLM-as-judge score for whether all claims in the answer are grounded in the retrieved context."""
        system_prompt = (
            "You are an expert fact-checking judge. Your task is to evaluate the Faithfulness of a generated "
            "answer compared to the provided search Context.\n\n"
            "Instructions:\n"
            "1. Check if all statements and claims in the Generated Answer are directly supported by, and can "
            "be inferred from, the provided Context.\n"
            "2. If the answer contains facts not found in the Context (external knowledge), it is unfaithful.\n"
            "3. If the answer states that it does not have enough information, and the Context indeed does not "
            "contain the answer, it is 100% faithful (1.0).\n"
            "4. Output a single numeric score between 0.0 and 1.0 (inclusive).\n"
            "5. Output ONLY the numeric float score (e.g., 1.0, 0.8, 0.0). Do NOT include any other text."
        )
        
        user_prompt = (
            f"Context:\n"
            f"\"\"\"\n"
            f"{retrieved_context.strip()}\n"
            f"\"\"\"\n\n"
            f"Generated Answer:\n"
            f"\"\"\"\n"
            f"{answer.strip()}\n"
            f"\"\"\"\n"
        )
        
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=5
            )
            score_str = (response.choices[0].message.content or "").strip()
            match = re.search(r"\d+(?:\.\d+)?", score_str)
            if match:
                score = float(match.group(0))
                return min(max(score, 0.0), 1.0)
            return 0.0
        except Exception as e:
            print(f"Error during faithfulness check: {e}")
            return 0.0

    def evaluate_retrieval_relevance(self, retrieved_chunks: List[Any], golden_sources: List[str]) -> float:
        """Measures whether the correct documents were retrieved (Hit Rate @ K)."""
        if not golden_sources:
            return 1.0
            
        retrieved_files = {res["chunk"].source_file for res in retrieved_chunks if "chunk" in res}
        hit = any(src in retrieved_files for src in golden_sources)
        return 1.0 if hit else 0.0

    def evaluate_citation_accuracy(self, verification_results: List[Dict[str, Any]]) -> float:
        """Measures what percentage of inline citations actually support the claims they are attached to."""
        if not verification_results:
            return 1.0
            
        verified_count = sum(1 for res in verification_results if res.get("result") == "VERIFIED")
        return float(verified_count / len(verification_results))

    def evaluate_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the pipeline and calculates all metrics for a single test case."""
        question = test_case["question"]
        golden_answer = test_case["ground_truth"]
        golden_sources = test_case["source_documents"]
        
        # 1. Retrieve
        retrieved_results = self.retriever.hybrid_search(question, k=5)
        
        # 2. Generate
        pipeline_output = self.generator.generate_response(question, retrieved_results)
        generated_answer = pipeline_output["answer"]
        verification_results = pipeline_output.get("verification_results", [])
        
        # Format context for faithfulness check
        chunks = [res["chunk"] for res in retrieved_results if "chunk" in res]
        formatted_context = self.generator.format_context_blocks(chunks)
        
        # 3. Calculate metrics
        correctness = self.evaluate_correctness(generated_answer, golden_answer)
        faithfulness = self.evaluate_faithfulness(generated_answer, formatted_context)
        retrieval_relevance = self.evaluate_retrieval_relevance(retrieved_results, golden_sources)
        citation_accuracy = self.evaluate_citation_accuracy(verification_results)
        
        return {
            "question": question,
            "generated_answer": generated_answer,
            "ground_truth": golden_answer,
            "category": test_case.get("category", "unknown"),
            "metrics": {
                "correctness": correctness,
                "faithfulness": faithfulness,
                "retrieval_relevance": retrieval_relevance,
                "citation_accuracy": citation_accuracy
            },
            "confidence_report": pipeline_output.get("confidence_report", {}),
            "fallback_triggered": pipeline_output.get("fallback_triggered", False)
        }

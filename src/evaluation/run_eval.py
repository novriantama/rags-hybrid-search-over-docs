import sys
import os
import json
import argparse
import re
import datetime
from pathlib import Path
from typing import List, Dict, Any, cast
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# pyrefly: ignore [missing-import]
from src.config import settings
# pyrefly: ignore [missing-import]
from src.evaluation.evaluator import RAGEvaluator

# List of representative question IDs for the --quick flag
QUICK_QUESTION_IDS = ["q1", "q8", "q26", "q41", "q48"]

class MockOpenAIClient:
    """Simulates OpenAI client responses for generation, verification, and evaluation judges."""
    def __init__(self):
        self.embeddings = MagicMock()
        self.chat = MagicMock()
        
        # Configure embeddings mock dynamically
        self.embeddings.create = self.mock_embeddings_create
        
        # Configure chat completions mock dynamically
        self.chat.completions.create = self.mock_chat_completion

    def mock_embeddings_create(self, input, model):
        mock_response = MagicMock()
        
        if isinstance(input, str):
            count = 1
        else:
            count = len(input)
            
        embs = []
        for _ in range(count):
            mock_emb = MagicMock()
            mock_emb.embedding = [0.0] * 1536
            embs.append(mock_emb)
            
        mock_response.data = embs
        return mock_response
        
    def mock_chat_completion(self, model, messages, temperature=0.0, max_tokens=None):
        system_content = messages[0]["content"] if messages else ""
        user_content = messages[1]["content"] if len(messages) > 1 else ""
        
        mock_response = MagicMock()
        mock_choice = MagicMock()
        
        system_prompt_lower = system_content.lower()
        
        if "quality assurance judge" in system_prompt_lower:
            # Correctness evaluation check
            mock_choice.message.content = " 0.94 "
        elif "fact-checking judge" in system_prompt_lower:
            # Faithfulness evaluation check
            mock_choice.message.content = " 1.0 "
        elif "fact-checking assistant" in system_prompt_lower:
            # Citation verification check
            mock_choice.message.content = "VERIFIED"
        elif "quality assurance assistant" in system_prompt_lower:
            # Completeness evaluation check
            mock_choice.message.content = " 0.98 "
        else:
            # Standard generation - return simulated answer based on question type
            question_match = re.search(r"Question:\s*(.*)", user_content)
            question = question_match.group(1).strip() if question_match else ""
            
            if "expiration" in question.lower() or "expire" in question.lower():
                mock_choice.message.content = "Access tokens expire exactly 1 hour after issuance [1]."
            elif "primary production" in question.lower():
                mock_choice.message.content = "The primary production database host is prod-db-01.internal.net [1]."
            elif "8001" in question.lower():
                mock_choice.message.content = "The Authentication Service runs on port 8001 [1]. Communicate with it using a Bearer token [2]."
            elif "timeout" in question.lower():
                mock_choice.message.content = "Database connections timeout after 10 minutes [1], and deployment pipeline rollbacks timeout after 3 minutes [2]."
            else:
                mock_choice.message.content = "I do not have enough information in the provided context to answer this question."
                
        mock_response.choices = [mock_choice]
        return mock_response

def parse_args():
    parser = argparse.ArgumentParser(description="Run the RAG automated evaluation suite.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only 5 representative test cases to save time and API costs."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit evaluation to the first N test cases."
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="./data/evaluation/reports",
        help="Directory to save evaluation report files."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force running evaluation in simulated mock mode without making API calls."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load golden Q&A dataset
    dataset_path = Path("./data/evaluation/golden_qa.json")
    if not dataset_path.exists():
        print(f"Error: Golden Q&A dataset not found at {dataset_path}")
        print("Please run Q&A dataset generation first.")
        sys.exit(1)
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)
        
    # Filter test cases based on arguments
    if args.quick:
        test_cases = [case for case in all_cases if case["id"] in QUICK_QUESTION_IDS]
        print(f"Running in --quick mode. Selected {len(test_cases)} representative test cases.")
    elif args.limit > 0:
        test_cases = all_cases[:args.limit]
        print(f"Limiting evaluation to first {len(test_cases)} test cases.")
    else:
        test_cases = all_cases
        print(f"Running evaluation on the complete dataset ({len(test_cases)} test cases).")
        
    # Check if we should use the Mock client
    is_mock_mode = args.mock or "your_openai" in settings.openai_api_key or not settings.openai_api_key
    
    if is_mock_mode:
        print("\n=== Running in MOCK mode to simulate evaluation without API calls ===")
        mock_client = MockOpenAIClient()
        
        # pyrefly: ignore [missing-import]
        from src.retrieval.dense import DenseIndex
        # pyrefly: ignore [missing-import]
        from src.retrieval.retriever import HybridRetriever
        # pyrefly: ignore [missing-import]
        from src.generation.generator import GroundedGenerator
        
        # We also pass a dummy directory for persistent db so it doesn't conflict
        dense_index = DenseIndex(openai_client=cast(Any, mock_client))
        retriever = HybridRetriever(dense_index=dense_index)
        generator = GroundedGenerator(openai_client=cast(Any, mock_client))
        evaluator = RAGEvaluator(retriever=retriever, generator=generator)
    else:
        print("\nRunning in real API mode.")
        evaluator = RAGEvaluator()
        
    results = []
    total = len(test_cases)
    
    print("\nExecuting evaluation runs:")
    for idx, case in enumerate(test_cases, 1):
        print(f"[{idx}/{total}] Question ID: {case['id']} | Category: {case['category']}")
        print(f"  Q: {case['question']}")
        
        try:
            res = evaluator.evaluate_case(case)
            results.append(res)
            metrics = res["metrics"]
            print(f"  -> Correctness: {metrics['correctness']:.2f} | Faithfulness: {metrics['faithfulness']:.2f} | Relevance: {metrics['retrieval_relevance']:.2f} | Citation: {metrics['citation_accuracy']:.2f}")
        except Exception as e:
            print(f"  ERROR evaluating question: {e}")
            
    if not results:
        print("Error: No test cases were evaluated.")
        sys.exit(1)
        
    # 3. Calculate Aggregates
    summary_report = calculate_summary(results)
    
    # 4. Save JSON Report
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = report_dir / f"eval_report_{timestamp}.json"
    
    report_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": summary_report,
        "detail": results
    }
    
    with open(report_filename, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
        
    print(f"\nDetailed evaluation report saved to: {report_filename}")
    
    # 5. Print beautiful console report
    print_results_table(summary_report)

def calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates average metric values overall and grouped by categories."""
    overall = {
        "correctness": 0.0,
        "faithfulness": 0.0,
        "retrieval_relevance": 0.0,
        "citation_accuracy": 0.0
    }
    
    by_category: Dict[str, Dict[str, Any]] = {}
    
    for item in results:
        m = item["metrics"]
        cat = item["category"]
        
        # Overall accumulation
        for k in overall:
            overall[k] += m[k]
            
        # Category accumulation
        if cat not in by_category:
            by_category[cat] = {
                "count": 0,
                "correctness": 0.0,
                "faithfulness": 0.0,
                "retrieval_relevance": 0.0,
                "citation_accuracy": 0.0
            }
        by_category[cat]["count"] += 1
        for k in overall:
            by_category[cat][k] += m[k]
            
    # Compute overall averages
    count = len(results)
    overall_avg = {k: round(v / count, 4) for k, v in overall.items()}
    
    # Compute category averages
    category_avg = {}
    for cat, data in by_category.items():
        cat_count = data["count"]
        category_avg[cat] = {
            "count": cat_count,
            "correctness": round(data["correctness"] / cat_count, 4),
            "faithfulness": round(data["faithfulness"] / cat_count, 4),
            "retrieval_relevance": round(data["retrieval_relevance"] / cat_count, 4),
            "citation_accuracy": round(data["citation_accuracy"] / cat_count, 4)
        }
        
    return {
        "total_cases": count,
        "overall_averages": overall_avg,
        "category_averages": category_avg
    }

def print_results_table(summary: Dict[str, Any]):
    """Outputs a clean, formatted ASCII table summarizing performance metrics."""
    print("\n" + "=" * 80)
    print("                    RAG PIPELINE EVALUATION SUMMARY REPORT                    ")
    print("=" * 80)
    print(f"Total Evaluated Cases: {summary['total_cases']}")
    print("-" * 80)
    
    headers = f"{'Category / Slice':<25} | {'Count':<5} | {'Correct':<7} | {'Faithful':<8} | {'RetrRel':<7} | {'CitAcc':<7}"
    print(headers)
    print("-" * 80)
    
    # Print overall
    o = summary["overall_averages"]
    overall_row = f"{'OVERALL AVERAGE':<25} | {summary['total_cases']:<5} | {o['correctness']:.4f} | {o['faithfulness']:.4f} | {o['retrieval_relevance']:.4f} | {o['citation_accuracy']:.4f}"
    print(overall_row)
    print("-" * 80)
    
    # Print categories
    for cat, metrics in summary["category_averages"].items():
        row = f"{cat:<25} | {metrics['count']:<5} | {metrics['correctness']:.4f} | {metrics['faithfulness']:.4f} | {metrics['retrieval_relevance']:.4f} | {metrics['citation_accuracy']:.4f}"
        print(row)
        
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

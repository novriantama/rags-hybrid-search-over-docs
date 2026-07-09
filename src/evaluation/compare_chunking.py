import sys
import os
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any, cast
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# pyrefly: ignore [missing-import]
from src.config import settings
# pyrefly: ignore [missing-import]
from src.ingestion.loader import DocumentLoader
# pyrefly: ignore [missing-import]
from src.ingestion.chunker import ChunkingOrchestrator
# pyrefly: ignore [missing-import]
from src.retrieval.index_manager import IndexManager
# pyrefly: ignore [missing-import]
from src.retrieval.retriever import HybridRetriever
# pyrefly: ignore [missing-import]
from src.generation.generator import GroundedGenerator
# pyrefly: ignore [missing-import]
from src.evaluation.evaluator import RAGEvaluator
# pyrefly: ignore [missing-import]
from src.evaluation.run_eval import MockOpenAIClient, QUICK_QUESTION_IDS

def parse_args():
    parser = argparse.ArgumentParser(description="Compare performance across chunking strategies.")
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
        help="Directory to save comparison report files."
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
        print(f"Running comparison in --quick mode with {len(test_cases)} test cases.")
    elif args.limit > 0:
        test_cases = all_cases[:args.limit]
        print(f"Limiting comparison evaluation to first {len(test_cases)} test cases.")
    else:
        test_cases = all_cases
        print(f"Running comparison evaluation on the complete dataset ({len(test_cases)} test cases).")
        
    # Check if we should use the Mock client
    is_mock_mode = args.mock or "your_openai" in settings.openai_api_key or not settings.openai_api_key
    
    # 2. Setup orchestration clients
    if is_mock_mode:
        print("\n=== Running in MOCK mode to simulate chunking comparison ===")
        mock_client = MockOpenAIClient()
        
        loader = DocumentLoader()
        orchestrator = ChunkingOrchestrator(openai_client=cast(Any, mock_client))
        manager = IndexManager(openai_client=cast(Any, mock_client))
        
        retriever = HybridRetriever(dense_index=manager.dense_index, sparse_index=manager.sparse_index)
        generator = GroundedGenerator(openai_client=cast(Any, mock_client))
        evaluator = RAGEvaluator(retriever=retriever, generator=generator)
    else:
        print("\nRunning comparison in real API mode.")
        loader = DocumentLoader()
        orchestrator = ChunkingOrchestrator()
        manager = IndexManager()
        evaluator = RAGEvaluator(
            retriever=HybridRetriever(dense_index=manager.dense_index, sparse_index=manager.sparse_index)
        )
        
    # 3. Load processed document fragments
    fragments = loader.load_processed_fragments()
    if not fragments:
        print("Error: No processed fragments found. Run ingestion first.")
        sys.exit(1)
        
    strategies = ["fixed", "recursive", "semantic"]
    comparison_results = {}
    
    for strategy in strategies:
        print("\n" + "=" * 60)
        print(f" EVALUATING STRATEGY: {strategy.upper()} ")
        print("=" * 60)
        
        # A. Clear indexes
        manager.clear_indexes()
        
        # B. Chunk fragments
        chunks = orchestrator.chunk_documents(fragments, strategy)
        print(f"Generated {len(chunks)} chunks using '{strategy}' strategy.")
        
        # C. Index chunks
        indexed, skipped = manager.index_chunks(chunks)
        print(f"Indexed: {indexed} | Skipped: {skipped} chunks.")
        
        # D. Run evaluation suite
        strategy_results = []
        total = len(test_cases)
        for idx, case in enumerate(test_cases, 1):
            print(f"  [{idx}/{total}] Question {case['id']}...")
            try:
                res = evaluator.evaluate_case(case)
                strategy_results.append(res)
            except Exception as e:
                print(f"    ERROR evaluating question: {e}")
                
        # E. Calculate aggregates
        if strategy_results:
            summary = calculate_averages(strategy_results, len(chunks))
            comparison_results[strategy] = {
                "chunk_count": len(chunks),
                "averages": summary,
                "detail": strategy_results
            }
            
    # 4. Print comparison report
    print_comparison_table(comparison_results)
    
    # 5. Save report
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_filename = report_dir / "chunking_comparison.json"
    
    comparison_report_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "mock_mode": is_mock_mode,
        "results": comparison_results
    }
    
    with open(report_filename, "w", encoding="utf-8") as f:
        json.dump(comparison_report_data, f, indent=2, ensure_ascii=False)
        
    print(f"Detailed comparison report saved to: {report_filename}\n")

def calculate_averages(results: List[Dict[str, Any]], chunk_count: int) -> Dict[str, float]:
    """Calculates overall metric averages for the run."""
    totals = {
        "correctness": 0.0,
        "faithfulness": 0.0,
        "retrieval_relevance": 0.0,
        "citation_accuracy": 0.0
    }
    count = len(results)
    for res in results:
        m = res["metrics"]
        for k in totals:
            totals[k] += m[k]
            
    return {k: round(v / count, 4) for k, v in totals.items()}

def print_comparison_table(results: Dict[str, Any]):
    """Outputs a clean ASCII comparative report table."""
    print("\n" + "=" * 85)
    print("                    CHUNKING STRATEGIES COMPARISON REPORT                    ")
    print("=" * 85)
    
    headers = f"{'Strategy':<15} | {'Chunks':<6} | {'Correctness':<11} | {'Faithfulness':<12} | {'RetrievalRel':<12} | {'CitationAcc':<11}"
    print(headers)
    print("-" * 85)
    
    for strategy, data in results.items():
        avg = data["averages"]
        row = (
            f"{strategy:<15} | {data['chunk_count']:<6} | "
            f"{avg['correctness']:.4f}     | {avg['faithfulness']:.4f}     | "
            f"{avg['retrieval_relevance']:.4f}     | {avg['citation_accuracy']:.4f}"
        )
        print(row)
        
    print("-" * 85)
    
    # Simple recommendation analyzer
    best_correctness = max(results.keys(), key=lambda x: results[x]["averages"]["correctness"])
    best_relevance = max(results.keys(), key=lambda x: results[x]["averages"]["retrieval_relevance"])
    
    print("\nArchitecture Drivers & Insights:")
    print(f"- Best Semantic Answer Correctness: '{best_correctness.upper()}' chunking.")
    print(f"- Best Retrieval Relevance (Hit Rate): '{best_relevance.upper()}' chunking.")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()

import pytest
# pyrefly: ignore [missing-import]
from src.evaluation.compare_chunking import calculate_averages, print_comparison_table

def test_calculate_averages():
    results = [
        {
            "metrics": {
                "correctness": 1.0,
                "faithfulness": 1.0,
                "retrieval_relevance": 1.0,
                "citation_accuracy": 1.0
            }
        },
        {
            "metrics": {
                "correctness": 0.8,
                "faithfulness": 0.9,
                "retrieval_relevance": 0.0,
                "citation_accuracy": 0.5
            }
        }
    ]
    
    averages = calculate_averages(results, chunk_count=10)
    
    # Expected averages:
    # Correctness: (1.0 + 0.8) / 2 = 0.9
    # Faithfulness: (1.0 + 0.9) / 2 = 0.95
    # Retrieval Relevance: (1.0 + 0.0) / 2 = 0.5
    # Citation Accuracy: (1.0 + 0.5) / 2 = 0.75
    assert averages["correctness"] == 0.9
    assert averages["faithfulness"] == 0.95
    assert averages["retrieval_relevance"] == 0.5
    assert averages["citation_accuracy"] == 0.75

def test_print_comparison_table(capsys):
    results = {
        "fixed": {
            "chunk_count": 20,
            "averages": {
                "correctness": 0.88,
                "faithfulness": 0.92,
                "retrieval_relevance": 0.80,
                "citation_accuracy": 0.85
            }
        },
        "recursive": {
            "chunk_count": 15,
            "averages": {
                "correctness": 0.94,
                "faithfulness": 0.98,
                "retrieval_relevance": 1.00,
                "citation_accuracy": 0.95
            }
        },
        "semantic": {
            "chunk_count": 12,
            "averages": {
                "correctness": 0.95,
                "faithfulness": 1.00,
                "retrieval_relevance": 1.00,
                "citation_accuracy": 1.00
            }
        }
    }
    
    print_comparison_table(results)
    captured = capsys.readouterr()
    
    # Verify key headers and values are printed
    assert "CHUNKING STRATEGIES COMPARISON REPORT" in captured.out
    assert "fixed" in captured.out
    assert "recursive" in captured.out
    assert "semantic" in captured.out
    assert "Best Semantic Answer Correctness: 'SEMANTIC' chunking." in captured.out

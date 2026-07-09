import pytest
from unittest.mock import MagicMock
# pyrefly: ignore [missing-import]
from src.ingestion.chunker import DocumentChunk
# pyrefly: ignore [missing-import]
from src.evaluation.evaluator import RAGEvaluator

def test_evaluate_correctness():
    # Mock OpenAI client
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = " 0.92 "
    mock_chat_completion = MagicMock()
    mock_chat_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_chat_completion

    # Create dummy generator and evaluator
    mock_generator = MagicMock()
    mock_generator.openai_client = mock_client
    
    evaluator = RAGEvaluator(generator=mock_generator)
    score = evaluator.evaluate_correctness("Access tokens expire in 1 hour.", "Tokens expire after 1 hour.")
    
    assert score == 0.92
    mock_client.chat.completions.create.assert_called_once()

def test_evaluate_faithfulness():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = " 1.0 "
    mock_chat_completion = MagicMock()
    mock_chat_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_chat_completion

    mock_generator = MagicMock()
    mock_generator.openai_client = mock_client
    
    evaluator = RAGEvaluator(generator=mock_generator)
    score = evaluator.evaluate_faithfulness("Answer content.", "Context content.")
    
    assert score == 1.0
    mock_client.chat.completions.create.assert_called_once()

def test_evaluate_retrieval_relevance():
    evaluator = RAGEvaluator()
    
    # Mock retrieved chunks
    chunk1 = DocumentChunk(
        id="c1", text="text", source_file="auth_service.md",
        section_heading="Auth", file_type="md", chunking_strategy="fixed", character_count=4
    )
    chunk2 = DocumentChunk(
        id="c2", text="text", source_file="database_guide.md",
        section_heading="DB", file_type="md", chunking_strategy="fixed", character_count=4
    )
    
    retrieved = [
        {"chunk": chunk1, "score": 0.9},
        {"chunk": chunk2, "score": 0.8}
    ]
    
    # Case 1: Hit (golden source is in retrieved files)
    assert evaluator.evaluate_retrieval_relevance(retrieved, ["auth_service.md"]) == 1.0
    
    # Case 2: Miss
    assert evaluator.evaluate_retrieval_relevance(retrieved, ["monitoring.md"]) == 0.0
    
    # Case 3: Empty golden (unanswerable) -> should return 1.0
    assert evaluator.evaluate_retrieval_relevance(retrieved, []) == 1.0

def test_evaluate_citation_accuracy():
    evaluator = RAGEvaluator()
    
    # Case 1: All verified
    verifications = [
        {"claim": "Claim 1", "citation_index": 1, "result": "VERIFIED"},
        {"claim": "Claim 2", "citation_index": 2, "result": "VERIFIED"}
    ]
    assert evaluator.evaluate_citation_accuracy(verifications) == 1.0
    
    # Case 2: Mixed
    verifications_mixed = [
        {"claim": "Claim 1", "citation_index": 1, "result": "VERIFIED"},
        {"claim": "Claim 2", "citation_index": 2, "result": "UNSUPPORTED"}
    ]
    assert evaluator.evaluate_citation_accuracy(verifications_mixed) == 0.5
    
    # Case 3: Empty (no citations) -> should return 1.0
    assert evaluator.evaluate_citation_accuracy([]) == 1.0

def test_evaluate_case():
    # Mock dependencies
    mock_retriever = MagicMock()
    mock_generator = MagicMock()
    mock_client = MagicMock()
    mock_generator.openai_client = mock_client
    
    # Mock hybrid search output
    chunk = DocumentChunk(
        id="c1", text="Server auth info.", source_file="auth_service.md",
        section_heading="Auth", file_type="md", chunking_strategy="fixed", character_count=17
    )
    mock_retrieved = [{"chunk": chunk, "score": 0.95}]
    mock_retriever.hybrid_search.return_value = mock_retrieved
    
    # Mock generator response
    mock_generator.format_context_blocks.return_value = "Formatted context blocks."
    mock_generator.generate_response.return_value = {
        "answer": "To authenticate, supply Bearer token [1].",
        "verification_results": [{"claim": "To authenticate, supply Bearer token.", "citation_index": 1, "result": "VERIFIED"}],
        "confidence_report": {"composite_score": 0.98},
        "fallback_triggered": False
    }
    
    # Mock LLM judge calls: 1st for correctness, 2nd for faithfulness
    mock_choice_correct = MagicMock()
    mock_choice_correct.message.content = " 0.95 "
    mock_choice_faithful = MagicMock()
    mock_choice_faithful.message.content = " 1.0 "
    
    mock_completion_correct = MagicMock()
    mock_completion_correct.choices = [mock_choice_correct]
    mock_completion_faithful = MagicMock()
    mock_completion_faithful.choices = [mock_choice_faithful]
    
    mock_client.chat.completions.create.side_effect = [
        mock_completion_correct,
        mock_completion_faithful
    ]
    
    evaluator = RAGEvaluator(retriever=mock_retriever, generator=mock_generator)
    
    test_case = {
        "id": "q1",
        "category": "straightforward_lookup",
        "question": "How to authenticate?",
        "ground_truth": "Supply Bearer token.",
        "source_documents": ["auth_service.md"]
    }
    
    res = evaluator.evaluate_case(test_case)
    
    assert res["question"] == "How to authenticate?"
    assert res["generated_answer"] == "To authenticate, supply Bearer token [1]."
    assert res["fallback_triggered"] is False
    
    metrics = res["metrics"]
    assert metrics["correctness"] == 0.95
    assert metrics["faithfulness"] == 1.0
    assert metrics["retrieval_relevance"] == 1.0
    assert metrics["citation_accuracy"] == 1.0
    
    mock_retriever.hybrid_search.assert_called_once_with("How to authenticate?", k=5)
    mock_generator.generate_response.assert_called_once_with("How to authenticate?", mock_retrieved)

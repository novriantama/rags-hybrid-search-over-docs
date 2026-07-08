import pytest
from unittest.mock import MagicMock
# pyrefly: ignore [missing-import]
from src.ingestion.chunker import DocumentChunk
# pyrefly: ignore [missing-import]
from src.generation.generator import GroundedGenerator

@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            id="doc1:sec1:0",
            text="The API server requires a valid token header for auth.",
            source_file="auth.md",
            section_heading="Authentication",
            page_number=1,
            file_type="md",
            chunking_strategy="fixed",
            character_count=54
        ),
        DocumentChunk(
            id="doc1:sec2:0",
            text="Database connection pool size defaults to 20 connections.",
            source_file="db.md",
            section_heading="Database Setup",
            page_number=3,
            file_type="md",
            chunking_strategy="fixed",
            character_count=56
        )
    ]

def test_format_context_blocks(sample_chunks):
    generator = GroundedGenerator()
    formatted = generator.format_context_blocks(sample_chunks)
    
    assert "Context Block [1]" in formatted
    assert "Source: auth.md (Section: Authentication) (Page: 1)" in formatted
    assert "Content: The API server requires a valid token header for auth." in formatted
    
    assert "Context Block [2]" in formatted
    assert "Source: db.md (Section: Database Setup) (Page: 3)" in formatted
    assert "Content: Database connection pool size defaults to 20 connections." in formatted

def test_format_context_blocks_empty():
    generator = GroundedGenerator()
    formatted = generator.format_context_blocks([])
    assert formatted == "(No context blocks available)"

def test_get_system_prompt():
    generator = GroundedGenerator()
    prompt = generator.get_system_prompt()
    assert "ONLY using the facts directly mentioned" in prompt
    assert "[1]" in prompt
    assert "I do not have enough information in the provided context to answer this question" in prompt

def test_get_fallback_system_prompt():
    generator = GroundedGenerator()
    prompt = generator.get_fallback_system_prompt()
    assert "retrieval confidence is below the threshold" in prompt
    assert "### What We Found" in prompt
    assert "### What Is Missing" in prompt
    assert "### Recommended Documents" in prompt

def test_parse_claims_and_citations():
    generator = GroundedGenerator()
    text = "The API server requires a valid token [1]. Database pool has 20 connections [2][1]."
    parsed = generator._parse_claims_and_citations(text)
    
    assert len(parsed) == 3
    
    # Sentence 1 has citation 1
    assert parsed[0]["claim"] == "The API server requires a valid token."
    assert parsed[0]["citation_index"] == 1
    
    # Sentence 2 has citation 2
    assert parsed[1]["claim"] == "Database pool has 20 connections."
    assert parsed[1]["citation_index"] == 2
    
    # Sentence 2 also has citation 1
    assert parsed[2]["claim"] == "Database pool has 20 connections."
    assert parsed[2]["citation_index"] == 1

def test_verify_citations(sample_chunks):
    mock_client = MagicMock()
    
    # Configure mock choices for successive calls (LLM-as-judge calls)
    mock_choice_verified = MagicMock()
    mock_choice_verified.message.content = "VERIFIED"
    
    mock_choice_unsupported = MagicMock()
    mock_choice_unsupported.message.content = "UNSUPPORTED"
    
    mock_chat_completion_1 = MagicMock()
    mock_chat_completion_1.choices = [mock_choice_verified]
    
    mock_chat_completion_2 = MagicMock()
    mock_chat_completion_2.choices = [mock_choice_unsupported]
    
    mock_client.chat.completions.create.side_effect = [
        mock_chat_completion_1,
        mock_chat_completion_2
    ]

    generator = GroundedGenerator(openai_client=mock_client)
    answer = "The API server requires a valid token [1]. Non-existent info goes here [2]."
    
    verified_results = generator.verify_citations(answer, sample_chunks)
    
    assert len(verified_results) == 2
    
    # Block [1] matches chunk 1 (auth.md) and was VERIFIED
    assert verified_results[0]["citation_index"] == 1
    assert verified_results[0]["source_file"] == "auth.md"
    assert verified_results[0]["result"] == "VERIFIED"
    
    # Block [2] matches chunk 2 (db.md) and was UNSUPPORTED
    assert verified_results[1]["citation_index"] == 2
    assert verified_results[1]["source_file"] == "db.md"
    assert verified_results[1]["result"] == "UNSUPPORTED"

def test_calculate_citation_coverage():
    generator = GroundedGenerator()
    answer = "Claims statement one [1]. Statement two has no citation. Statement three has verified [2]."
    
    verification_results = [
        {"claim": "Claims statement one.", "citation_index": 1, "result": "VERIFIED"},
        {"claim": "Statement three has verified.", "citation_index": 2, "result": "VERIFIED"}
    ]
    
    # Sentence 1 is verified (has citation, VERIFIED)
    # Sentence 2 has no citation (so not in verification_results) -> not verified
    # Sentence 3 is verified (has citation, VERIFIED)
    # Total sentences = 3, verified = 2. Coverage = 2/3
    coverage = generator.calculate_citation_coverage(answer, verification_results)
    assert abs(coverage - 0.6666) < 0.01

    # Test with partially unsupported citation
    verification_results_unsupported = [
        {"claim": "Claims statement one.", "citation_index": 1, "result": "VERIFIED"},
        {"claim": "Statement three has verified.", "citation_index": 2, "result": "UNSUPPORTED"}
    ]
    # Total sentences = 3, verified = 1 (sentence 1). Coverage = 1/3
    coverage_unsupported = generator.calculate_citation_coverage(answer, verification_results_unsupported)
    assert abs(coverage_unsupported - 0.3333) < 0.01

def test_evaluate_completeness():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = " 0.85 "
    mock_chat_completion = MagicMock()
    mock_chat_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_chat_completion

    generator = GroundedGenerator(openai_client=mock_client)
    score = generator._evaluate_completeness("What is port 5432 used for?", "Database uses port 5432.")
    assert score == 0.85

def test_calculate_confidence_score():
    generator = GroundedGenerator()
    scores = generator.calculate_confidence_score(
        retrieval_score=0.9,
        citation_coverage=0.8,
        completeness_score=0.95
    )
    expected = 0.33 * 0.9 + 0.33 * 0.8 + 0.34 * 0.95
    assert scores["composite_score"] == round(expected, 4)
    assert scores["retrieval_confidence"] == 0.9
    assert scores["citation_coverage"] == 0.8
    assert scores["answer_completeness"] == 0.95

def test_generate_response_success(sample_chunks):
    # Mock OpenAI client
    mock_client = MagicMock()
    
    # 1st response: generation
    mock_choice_gen = MagicMock()
    mock_choice_gen.message.content = "To authenticate, you must use a valid token header [1]."
    mock_chat_completion_gen = MagicMock()
    mock_chat_completion_gen.choices = [mock_choice_gen]
    
    # 2nd response: verification check (LLM-as-judge)
    mock_choice_ver = MagicMock()
    mock_choice_ver.message.content = "VERIFIED"
    mock_chat_completion_ver = MagicMock()
    mock_chat_completion_ver.choices = [mock_choice_ver]
    
    # 3rd response: completeness check (LLM-as-judge)
    mock_choice_comp = MagicMock()
    mock_choice_comp.message.content = " 1.0 "
    mock_chat_completion_comp = MagicMock()
    mock_chat_completion_comp.choices = [mock_choice_comp]
    
    mock_client.chat.completions.create.side_effect = [
        mock_chat_completion_gen,
        mock_chat_completion_ver,
        mock_chat_completion_comp
    ]

    generator = GroundedGenerator(openai_client=mock_client)
    
    # Pass chunk dicts with retrieval scores
    results_input = [
        {"chunk": sample_chunks[0], "score": 0.95},
        {"chunk": sample_chunks[1], "score": 0.8}
    ]
    
    res = generator.generate_response(
        question="How do I authenticate with the API?",
        chunks_or_results=results_input
    )

    assert res["answer"] == "To authenticate, you must use a valid token header [1]."
    assert "auth.md" in res["user_prompt"]
    assert res["fallback_triggered"] is False
    
    # Verify citations verification
    assert len(res["verification_results"]) == 1
    assert res["verification_results"][0]["result"] == "VERIFIED"
    
    # Verify confidence report
    report = res["confidence_report"]
    assert report["retrieval_confidence"] == 0.95  # only chunk 1 was cited, so its score is used
    assert report["citation_coverage"] == 1.0
    assert report["answer_completeness"] == 1.0
    assert report["composite_score"] == round(0.33 * 0.95 + 0.33 * 1.0 + 0.34 * 1.0, 4)

    # Verify calls
    assert mock_client.chat.completions.create.call_count == 3

def test_generate_response_insufficient_info(sample_chunks):
    mock_client = MagicMock()
    
    mock_choice_gen = MagicMock()
    mock_choice_gen.message.content = "I do not have enough information in the provided context to answer this question."
    mock_chat_completion_gen = MagicMock()
    mock_chat_completion_gen.choices = [mock_choice_gen]
    
    mock_choice_comp = MagicMock()
    mock_choice_comp.message.content = "1.0"
    mock_chat_completion_comp = MagicMock()
    mock_chat_completion_comp.choices = [mock_choice_comp]
    
    mock_client.chat.completions.create.side_effect = [
        mock_chat_completion_gen,
        mock_chat_completion_comp
    ]

    generator = GroundedGenerator(openai_client=mock_client)
    res = generator.generate_response(
        question="What is the default port for Postgres?",
        chunks_or_results=sample_chunks
    )

    assert res["answer"] == "I do not have enough information in the provided context to answer this question."
    assert mock_client.chat.completions.create.call_count == 2
    assert res["fallback_triggered"] is False

def test_generate_response_low_confidence(sample_chunks):
    # Mock OpenAI client
    mock_client = MagicMock()
    
    # 1st response: fallback generation report
    mock_choice_gen = MagicMock()
    mock_choice_gen.message.content = "### What We Found\nSome details.\n### What Is Missing\nEverything else.\n### Recommended Documents\nauth.md"
    mock_chat_completion_gen = MagicMock()
    mock_chat_completion_gen.choices = [mock_choice_gen]
    
    # 2nd response: completeness check
    mock_choice_comp = MagicMock()
    mock_choice_comp.message.content = " 0.8 "
    mock_chat_completion_comp = MagicMock()
    mock_chat_completion_comp.choices = [mock_choice_comp]
    
    mock_client.chat.completions.create.side_effect = [
        mock_chat_completion_gen,
        mock_chat_completion_comp
    ]

    generator = GroundedGenerator(openai_client=mock_client)
    
    # Chunks with low retrieval score (e.g. 0.5)
    results_input = [
        {"chunk": sample_chunks[0], "score": 0.5},
        {"chunk": sample_chunks[1], "score": 0.4}
    ]
    
    res = generator.generate_response(
        question="How do I setup a Redis database?",
        chunks_or_results=results_input,
        retrieval_threshold=0.7
    )

    assert "### What We Found" in res["answer"]
    assert "### What Is Missing" in res["answer"]
    assert res["fallback_triggered"] is True
    assert res["verification_results"] == []
    
    # Verify confidence report
    report = res["confidence_report"]
    assert report["retrieval_confidence"] == 0.5 # max score of chunks
    assert report["citation_coverage"] == 1.0 # default fallback coverage
    assert report["answer_completeness"] == 0.8
    assert report["composite_score"] == round(0.33 * 0.5 + 0.33 * 1.0 + 0.34 * 0.8, 4)

    # Verify calls
    assert mock_client.chat.completions.create.call_count == 2

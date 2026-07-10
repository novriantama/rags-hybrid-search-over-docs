import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# pyrefly: ignore [missing-import]
from src.api.main import app, retriever, generator, loader, orchestrator, manager

client = TestClient(app)

def test_ask_endpoint():
    # Mock retriever output
    mock_chunk = MagicMock()
    mock_chunk.id = "c1"
    mock_chunk.text = "Mock context text."
    mock_chunk.source_file = "auth_service.md"
    mock_chunk.section_heading = "Auth"
    mock_chunk.file_type = "md"
    mock_chunk.character_count = 18
    
    retriever.hybrid_search = MagicMock(return_value=[{"chunk": mock_chunk, "score": 0.98}])
    
    # Mock generator output
    generator.generate_response = MagicMock(return_value={
        "answer": "Test answer with citation [1].",
        "verification_results": [
            {
                "claim": "Test answer with citation.",
                "citation_index": 1,
                "source_file": "auth_service.md",
                "result": "VERIFIED"
            }
        ],
        "confidence_report": {
            "composite_score": 0.96,
            "retrieval_confidence": 0.98,
            "citation_coverage": 1.0,
            "completeness_score": 0.95
        },
        "fallback_triggered": False
    })
    
    payload = {
        "question": "What is the token expiration?",
        "threshold": 0.7,
        "chunking_strategy": "recursive",
        "dense_weight": 0.5,
        "sparse_weight": 0.5,
        "k": 5
    }
    
    response = client.post("/v1/ask", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "Test answer with citation [1]."
    assert data["confidence_report"]["composite_score"] == 0.96
    assert data["citations"][0]["result"] == "VERIFIED"
    assert data["fallback_triggered"] is False
    
    retriever.hybrid_search.assert_called_once_with(
        query="What is the token expiration?",
        k=5,
        dense_weight=0.5,
        sparse_weight=0.5,
        use_reranker=True
    )
    generator.generate_response.assert_called_once()

def test_documents_endpoint(tmp_path):
    # Save original paths to restore later
    orig_raw = loader.raw_dir
    orig_processed = loader.processed_dir
    
    try:
        # Mock raw_dir and processed_dir on the global loader instance
        loader.raw_dir = tmp_path / "raw"
        loader.processed_dir = tmp_path / "processed"
        
        loader.raw_dir.mkdir()
        loader.processed_dir.mkdir()
        
        # Create a dummy raw file
        raw_file = loader.raw_dir / "test_doc.md"
        raw_file.write_text("Hello world")
        
        # Create corresponding processed file
        processed_file = loader.processed_dir / "test_doc.json"
        processed_file.write_text(json.dumps({
            "source_file": "test_doc.md",
            "fragments": [{"text": "Hello world"}]
        }))
        
        response = client.get("/v1/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["filename"] == "test_doc.md"
        assert data[0]["fragment_count"] == 1
        assert data[0]["size_bytes"] == len("Hello world")
    finally:
        # Restore original paths
        loader.raw_dir = orig_raw
        loader.processed_dir = orig_processed

def test_ingest_endpoint(tmp_path):
    # Save original loader file ingestion and paths
    orig_ingest_file = loader.ingest_file
    orig_raw = loader.raw_dir
    orig_chunk_docs = orchestrator.chunk_documents
    orig_index_chunks = manager.index_chunks
    
    try:
        loader.raw_dir = tmp_path
        loader.ingest_file = MagicMock(return_value=[MagicMock()])
        orchestrator.chunk_documents = MagicMock(return_value=[MagicMock()])
        manager.index_chunks = MagicMock(return_value=(1, 0))
        
        # Simulate file upload
        file_content = b"Mock document content"
        files = {"file": ("uploaded_file.md", file_content, "text/plain")}
        
        response = client.post("/v1/ingest?chunking_strategy=fixed", files=files)
            
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "uploaded_file.md"
        assert data["status"] == "SUCCESS"
        assert data["chunks_indexed"] == 1
    finally:
        loader.ingest_file = orig_ingest_file
        loader.raw_dir = orig_raw
        orchestrator.chunk_documents = orig_chunk_docs
        manager.index_chunks = orig_index_chunks

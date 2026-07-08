import pytest
from unittest.mock import MagicMock
# pyrefly: ignore [missing-import]
from src.ingestion.chunker import DocumentChunk
# pyrefly: ignore [missing-import]
from src.retrieval.dense import DenseIndex
# pyrefly: ignore [missing-import]
from src.retrieval.sparse import SparseIndex
# pyrefly: ignore [missing-import]
from src.retrieval.retriever import HybridRetriever
# pyrefly: ignore [missing-import]
from src.retrieval.reranker import CrossEncoderReranker

@pytest.fixture
def sample_candidates():
    return [
        DocumentChunk(id="A", text="Authentication bearer token security", source_file="doc.md", file_type="md", chunking_strategy="fixed", character_count=36),
        DocumentChunk(id="B", text="Postgres microservices database port configuration", source_file="doc.md", file_type="md", chunking_strategy="fixed", character_count=50),
        DocumentChunk(id="C", text="File upload gateway service size limit settings", source_file="doc.md", file_type="md", chunking_strategy="fixed", character_count=47)
    ]

def test_jaccard_fallback_reranker(sample_candidates):
    # Enforce fallback mode so we don't load sentence-transformers model
    reranker = CrossEncoderReranker(force_fallback=True)
    
    # Query matching chunk A terms
    query = "bearer security token"
    results = reranker.rerank(query, sample_candidates, top_k=2)
    
    assert len(results) == 2
    # Chunk A has highest word overlap -> should be first
    assert results[0]["chunk"].id == "A"
    assert results[0]["score"] > 0.0
    
    # Chunk B and C have 0 overlap -> score should be 0.0
    assert results[1]["score"] == 0.0

def test_retriever_hybrid_search_with_rerank(sample_candidates):
    # Mock index managers
    mock_dense = MagicMock(spec=DenseIndex)
    mock_sparse = MagicMock(spec=SparseIndex)
    
    # Setup mock retriever searches (returning A then B)
    mock_dense.generate_embeddings.return_value = [[0.1] * 1536]
    mock_dense.search.return_value = [
        {
            "id": sample_candidates[0].id,
            "text": sample_candidates[0].text,
            "distance": 0.1,
            "metadata": {
                "source_file": sample_candidates[0].source_file,
                "section_heading": sample_candidates[0].section_heading,
                "chunking_strategy": sample_candidates[0].chunking_strategy,
                "character_count": sample_candidates[0].character_count,
                "file_type": sample_candidates[0].file_type
            }
        },
        {
            "id": sample_candidates[1].id,
            "text": sample_candidates[1].text,
            "distance": 0.2,
            "metadata": {
                "source_file": sample_candidates[1].source_file,
                "section_heading": sample_candidates[1].section_heading,
                "chunking_strategy": sample_candidates[1].chunking_strategy,
                "character_count": sample_candidates[1].character_count,
                "file_type": sample_candidates[1].file_type
            }
        }
    ]
    mock_sparse.search.return_value = [
        {"chunk": sample_candidates[0], "score": 2.0},
        {"chunk": sample_candidates[1], "score": 1.0}
    ]
    
    retriever = HybridRetriever(dense_index=mock_dense, sparse_index=mock_sparse)
    
    # Mock the reranker
    mock_reranker = MagicMock(spec=CrossEncoderReranker)
    # Reranker says B is better than A for this specific query
    mock_reranker.rerank.return_value = [
        {"chunk": sample_candidates[1], "score": 0.95},
        {"chunk": sample_candidates[0], "score": 0.45}
    ]
    retriever.reranker = mock_reranker
    
    # Executing hybrid search with reranking
    results = retriever.hybrid_search("Postgres config query", k=2, use_reranker=True)
    
    assert len(results) == 2
    # Should follow the reranked order: B then A
    assert results[0]["chunk"].id == "B"
    assert results[1]["chunk"].id == "A"
    assert results[0]["score"] == 0.95
    
    # Verify reranker was invoked with candidate list
    mock_reranker.rerank.assert_called_once()

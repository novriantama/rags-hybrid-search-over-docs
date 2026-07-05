import os
import pytest
from unittest.mock import MagicMock
from src.ingestion.chunker import DocumentChunk
from src.retrieval.dense import DenseIndex
from src.retrieval.sparse import SparseIndex
from src.retrieval.index_manager import IndexManager

@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            id="doc1:sec1:strategy:0",
            text="The authentication service requires a Bearer token in the Authorization header.",
            source_file="doc1.md",
            section_heading="Auth",
            page_number=1,
            file_type="md",
            chunking_strategy="recursive",
            character_count=80
        ),
        DocumentChunk(
            id="doc1:sec2:strategy:0",
            text="Production databases are hosted on Postgres at port 5432. Do not expose externally.",
            source_file="doc1.md",
            section_heading="DB",
            page_number=1,
            file_type="md",
            chunking_strategy="recursive",
            character_count=85
        )
    ]

def test_dense_index_operations(tmp_path, sample_chunks):
    persist_dir = tmp_path / "chroma"
    
    # Mock OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    emb1 = MagicMock()
    emb1.embedding = [1.0] + [0.0] * 1535
    emb2 = MagicMock()
    emb2.embedding = [0.0, 1.0] + [0.0] * 1534
    mock_response.data = [emb1, emb2]
    mock_client.embeddings.create.return_value = mock_response

    dense = DenseIndex(persist_directory=str(persist_dir), openai_client=mock_client)
    
    # Check embedding generation
    embs = dense.generate_embeddings(["text1", "text2"])
    assert len(embs) == 2
    assert embs[0] == [1.0] + [0.0] * 1535
    
    # Test add chunks
    dense.add_chunks(sample_chunks, embs)
    assert dense.collection.count() == 2
    
    # Test search query
    results = dense.search(query_embedding=embs[0], k=1)
    assert len(results) == 1
    assert results[0]["id"] == sample_chunks[0].id
    assert "authentication" in results[0]["text"]

def test_sparse_index_operations(tmp_path, sample_chunks):
    persist_file = tmp_path / "bm25" / "bm25_index.pkl"
    sparse = SparseIndex(persist_path=str(persist_file))
    
    # Add a third dummy chunk to make corpus size N=3, ensuring a positive IDF score in BM25
    dummy_chunk = DocumentChunk(
        id="doc1:sec3:strategy:0",
        text="This is a third dummy document to prevent zero/negative IDF scores for 50 percent terms.",
        source_file="doc1.md",
        section_heading="Dummy",
        page_number=1,
        file_type="md",
        chunking_strategy="recursive",
        character_count=88
    )
    corpus = sample_chunks + [dummy_chunk]
    
    sparse.add_chunks(corpus)
    assert len(sparse.chunks) == 3
    
    # Search for term
    results = sparse.search("Postgres")
    assert len(results) == 1
    assert results[0]["chunk"].id == sample_chunks[1].id
    assert results[0]["score"] > 0.0
    
    # Verify persistence reloading
    sparse2 = SparseIndex(persist_path=str(persist_file))
    assert len(sparse2.chunks) == 3
    assert sparse2.bm25 is not None

def test_index_manager_deduplication(tmp_path, sample_chunks):
    persist_chroma = tmp_path / "chroma"
    persist_bm25 = tmp_path / "bm25" / "bm25_index.pkl"
    
    # Mock OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    
    # Return 3 embeddings (for chunk 1, chunk 2, and a duplicate of chunk 1)
    emb1 = MagicMock()
    emb1.embedding = [1.0, 0.0, 0.0]
    emb2 = MagicMock()
    emb2.embedding = [0.0, 1.0, 0.0]
    emb3 = MagicMock()
    # Almost identical to emb1 (cosine similarity = 0.99)
    emb3.embedding = [0.99, 0.01, 0.0]
    
    mock_response.data = [emb1, emb2, emb3]
    mock_client.embeddings.create.return_value = mock_response
    
    dense = DenseIndex(persist_directory=str(persist_chroma), openai_client=mock_client)
    sparse = SparseIndex(persist_path=str(persist_bm25))
    
    manager = IndexManager(dense_index=dense, sparse_index=sparse, openai_client=mock_client)
    
    # We will pass 3 chunks where the 3rd chunk is a duplicate of the 1st
    chunks_to_index = sample_chunks + [
        DocumentChunk(
            id="doc1:sec1:strategy:1",
            text="The authentication service requires a Bearer token in the Authorization header. (Duplicate content)",
            source_file="doc1.md",
            section_heading="Auth",
            page_number=1,
            file_type="md",
            chunking_strategy="recursive",
            character_count=98
        )
    ]
    
    # Ingest the batch
    indexed_count, skipped_count = manager.index_chunks(chunks_to_index)
    
    assert indexed_count == 2
    assert skipped_count == 1
    
    # Check ChromaDB collection has only 2 items
    assert dense.collection.count() == 2
    
    # Check BM25 has only 2 items
    assert len(sparse.chunks) == 2
    
    # Check indexing duplicate in a subsequent batch
    # We will try to index the same duplicate again
    mock_response_subsequent = MagicMock()
    emb_dup = MagicMock()
    emb_dup.embedding = [1.0, 0.0, 0.0]
    mock_response_subsequent.data = [emb_dup]
    mock_client.embeddings.create.return_value = mock_response_subsequent
    
    indexed_count2, skipped_count2 = manager.index_chunks([chunks_to_index[2]])
    assert indexed_count2 == 0
    assert skipped_count2 == 1

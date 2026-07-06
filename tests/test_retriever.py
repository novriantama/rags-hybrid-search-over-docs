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

@pytest.fixture
def indexed_chunks():
    return [
        DocumentChunk(
            id="doc1:sec1:strategy:0",
            text="The API server requires a valid token header for auth.",
            source_file="doc1.md",
            section_heading="Authentication",
            page_number=1,
            file_type="md",
            chunking_strategy="recursive",
            character_count=54
        ),
        DocumentChunk(
            id="doc1:sec2:strategy:0",
            text="Postgres databases run on port 5432.",
            source_file="doc1.md",
            section_heading="Database",
            page_number=1,
            file_type="md",
            chunking_strategy="recursive",
            character_count=36
        )
    ]

def test_dense_search_retrieval(tmp_path, indexed_chunks):
    persist_dir = tmp_path / "chroma"
    
    # Mock OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    
    # Setup mock embeddings
    emb_doc1 = [1.0, 0.0, 0.0]
    emb_doc2 = [0.0, 1.0, 0.0]
    emb_query = [0.99, 0.01, 0.0] # highly similar to emb_doc1
    
    # Mock response during add_chunks and query (exactly 2 for corpus size)
    mock_response.data = [
        MagicMock(embedding=emb_doc1),
        MagicMock(embedding=emb_doc2)
    ]
    mock_client.embeddings.create.return_value = mock_response
    
    # Build indexes
    dense = DenseIndex(persist_directory=str(persist_dir), openai_client=mock_client)
    sparse = SparseIndex(persist_path=str(tmp_path / "bm25_index.pkl"))
    
    # Seed mock embeddings to populate DB
    # Generate embeddings for corpus (will return emb_doc1 and emb_doc2)
    embs = dense.generate_embeddings(["API", "Postgres"])
    dense.add_chunks(indexed_chunks, embs)
    
    # Re-mock embeddings create response for query execution (will return emb_query)
    mock_query_response = MagicMock()
    mock_query_response.data = [MagicMock(embedding=emb_query)]
    mock_client.embeddings.create.return_value = mock_query_response
    
    # Initialize retriever
    retriever = HybridRetriever(dense_index=dense, sparse_index=sparse)
    
    # Search
    results = retriever.dense_search("Query auth token", k=1)
    
    assert len(results) == 1
    res = results[0]
    
    # Verify score conversion is correct (distance is 1 - similarity)
    # Cosine Similarity of query [0.99, 0.01, 0.0] with doc1 [1.0, 0.0, 0.0] is 0.99
    # Distance in Chroma should be 1 - 0.99 = 0.01
    # Retrieved score should be 1.0 - 0.01 = 0.99
    assert res["score"] == pytest.approx(0.99, abs=1e-2)
    
    # Verify DocumentChunk conversion
    chunk = res["chunk"]
    assert isinstance(chunk, DocumentChunk)
    assert chunk.id == indexed_chunks[0].id
    assert chunk.text == indexed_chunks[0].text
    assert chunk.section_heading == "Authentication"
    assert chunk.source_file == "doc1.md"
    assert chunk.page_number == 1

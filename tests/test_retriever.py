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

def test_sparse_search_retrieval(tmp_path, indexed_chunks):
    persist_file = tmp_path / "bm25_index.pkl"
    sparse = SparseIndex(persist_path=str(persist_file))
    
    # Add a third dummy chunk to ensure positive BM25 scores
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
    corpus = indexed_chunks + [dummy_chunk]
    sparse.add_chunks(corpus)
    
    # Setup retriever (dense index is mocked since not used here)
    retriever = HybridRetriever(dense_index=MagicMock(), sparse_index=sparse)
    
    # Query for "Postgres" keyword, which should match doc2
    results = retriever.sparse_search("Postgres", k=1)
    
    assert len(results) == 1
    assert results[0]["chunk"].id == indexed_chunks[1].id
    assert results[0]["score"] > 0.0

def test_rrf_scoring():
    chunk_a = DocumentChunk(id="A", text="Text A", source_file="doc.md", file_type="md", chunking_strategy="fixed", character_count=6)
    chunk_b = DocumentChunk(id="B", text="Text B", source_file="doc.md", file_type="md", chunking_strategy="fixed", character_count=6)
    chunk_c = DocumentChunk(id="C", text="Text C", source_file="doc.md", file_type="md", chunking_strategy="fixed", character_count=6)
    
    # Dense results: A is 1st, B is 2nd
    dense_results = [
        {"chunk": chunk_a, "score": 0.9},
        {"chunk": chunk_b, "score": 0.8}
    ]
    # Sparse results: C is 1st, B is 2nd
    sparse_results = [
        {"chunk": chunk_c, "score": 1.5},
        {"chunk": chunk_b, "score": 1.2}
    ]
    
    retriever = HybridRetriever(dense_index=MagicMock(), sparse_index=MagicMock())
    
    # 1. Equal weights (0.5 dense / 0.5 sparse), k = 60
    # Score A = 0.5 * (1 / 61) = 0.008196
    # Score C = 0.5 * (1 / 61) = 0.008196
    # Score B = 0.5 * (1 / 62) + 0.5 * (1 / 62) = 1/62 = 0.016129
    # B should rank 1st because it appears in both lists, even though A & C were rank 1.
    fused_1 = retriever.reciprocal_rank_fusion(
        dense_results, sparse_results, dense_weight=0.5, sparse_weight=0.5, rrf_k=60
    )
    
    assert len(fused_1) == 3
    assert fused_1[0]["chunk"].id == "B" # boosted!
    assert fused_1[0]["score"] == pytest.approx(1.0 / 62.0)
    
    # 2. Configurable weights: Dense-heavy (0.9 dense / 0.1 sparse)
    # Score A = 0.9 * (1/61) = 0.014754
    # Score B = 0.9 * (1/62) + 0.1 * (1/62) = 1/62 = 0.016129
    # Score C = 0.1 * (1/61) = 0.001639
    # B still ranks 1st.
    # If we do Dense-only (1.0 dense / 0.0 sparse):
    # Score A = 1.0 * (1/61) = 0.016393
    # Score B = 1.0 * (1/62) = 0.016129
    # Score C = 0.0
    # A should rank 1st.
    fused_2 = retriever.reciprocal_rank_fusion(
        dense_results, sparse_results, dense_weight=1.0, sparse_weight=0.0, rrf_k=60
    )
    assert fused_2[0]["chunk"].id == "A"
    assert fused_2[1]["chunk"].id == "B"
    assert len(fused_2) == 2 # C is excluded because sparse weight is 0.0 (and didn't appear in dense)


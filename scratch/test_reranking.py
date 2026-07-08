import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent))

# pyrefly: ignore [missing-import]
from src.ingestion.loader import DocumentLoader
# pyrefly: ignore [missing-import]
from src.ingestion.chunker import ChunkingOrchestrator
# pyrefly: ignore [missing-import]
from src.retrieval.index_manager import IndexManager
# pyrefly: ignore [missing-import]
from src.retrieval.retriever import HybridRetriever

def main():
    print("=== Hybrid RAG Cross-Encoder Reranking Integration Test ===")
    
    # 1. Load sample document
    loader = DocumentLoader()
    fragments = loader.load_processed_fragments()
    if not fragments:
        print("Error: No processed fragments found. Run ingestion runner first.")
        return
        
    orchestrator = ChunkingOrchestrator()
    chunks = orchestrator.chunk_documents(fragments, "recursive")
    
    manager = IndexManager()
    if manager.dense_index.collection.count() == 0:
        print("Populating databases...")
        manager.index_chunks(chunks)
        
    # 2. Query configurations
    retriever = HybridRetriever(dense_index=manager.dense_index, sparse_index=manager.sparse_index)
    query = "Authorization Bearer microservices token timeout limits"
    
    # Run searches side-by-side
    print(f"\nQuery: '{query}'\n")
    
    # A. Search without Reranker (Top 3 RRF results)
    print("=== [1] Candidates before Re-ranking (RRF Top 3) ===")
    results_rrf = retriever.hybrid_search(
        query=query,
        k=3,
        use_reranker=False
    )
    for i, res in enumerate(results_rrf):
        print(f"  [{i+1}] RRF Score: {res['score']:.6f} | Heading: {res['chunk'].section_heading}")
        print(f"      Text: {res['chunk'].text[:100]}...")
    print()
    
    # B. Search with Reranker (RRF Top 20 Candidates -> Neural Reranked Top 3)
    # We will enforce the Jaccard fallback mode for the reranker in this manual runner to prevent
    # downloading a multi-hundred megabyte Hugging Face model during standard test suite execution.
    # To demonstrate normal retrieval, this fallback still refines order using keyword overlap!
    print("=== [2] Refined Candidates after Re-ranking (Top 3) ===")
    
    # Override retriever's reranker with force_fallback to prevent HF download
    # pyrefly: ignore [missing-import]
    from src.retrieval.reranker import CrossEncoderReranker
    retriever.reranker = CrossEncoderReranker(force_fallback=True)
    
    results_reranked = retriever.hybrid_search(
        query=query,
        k=3,
        use_reranker=True,
        candidate_k=20
    )
    for i, res in enumerate(results_reranked):
        print(f"  [{i+1}] Rerank Score: {res['score']:.6f} | Heading: {res['chunk'].section_heading}")
        print(f"      Text: {res['chunk'].text[:100]}...")
    print()

if __name__ == "__main__":
    main()

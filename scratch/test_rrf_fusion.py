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
    print("=== Hybrid RAG RRF Fusion Ingestion & Search Integration Test ===")
    
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
    
    # Run tests under different weights
    scenarios = [
        {"name": "Dense-only", "dense": 1.0, "sparse": 0.0},
        {"name": "Sparse-only", "dense": 0.0, "sparse": 1.0},
        {"name": "Balanced Hybrid", "dense": 0.5, "sparse": 0.5},
        {"name": "Dense-heavy Hybrid", "dense": 0.8, "sparse": 0.2}
    ]
    
    print(f"\nQuery: '{query}'\n")
    
    for scenario in scenarios:
        print(f"--- Scenario: {scenario['name']} (Dense weight: {scenario['dense']}, Sparse weight: {scenario['sparse']}) ---")
        results = retriever.hybrid_search(
            query=query,
            k=3,
            dense_weight=scenario['dense'],
            sparse_weight=scenario['sparse']
        )
        
        if not results:
            print("  No results returned.")
            
        for i, res in enumerate(results):
            print(f"  [{i+1}] Score: {res['score']:.6f}")
            print(f"      Chunk ID: {res['chunk'].id}")
            print(f"      Section heading: {res['chunk'].section_heading}")
            print(f"      Text: {res['chunk'].text[:100]}...")
        print()

if __name__ == "__main__":
    main()

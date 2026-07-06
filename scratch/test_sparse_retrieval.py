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
    print("=== Hybrid RAG Sparse Retrieval Integration Test ===")
    
    # 1. Check if we have documents indexed, if not index them first
    loader = DocumentLoader()
    fragments = loader.load_processed_fragments()
    if not fragments:
        print("Error: No processed fragments found. Run ingestion tests first.")
        return
        
    orchestrator = ChunkingOrchestrator()
    chunks = orchestrator.chunk_documents(fragments, "recursive")
    
    manager = IndexManager()
    # Check if empty, populate if needed
    if len(manager.sparse_index.chunks) == 0:
        print("Database is empty, indexing sample chunks first...")
        manager.index_chunks(chunks)
        
    # 2. Query the BM25 sparse search database
    retriever = HybridRetriever(dense_index=manager.dense_index, sparse_index=manager.sparse_index)
    
    # Execute query
    query = "Postgres databases hosted verify-full"
    print(f"\nExecuting Sparse Search for query: '{query}'")
    
    results = retriever.sparse_search(query, k=10)
    print(f"Retrieved {len(results)} chunks (ordered by BM25 score descending):\n")
    
    for i, res in enumerate(results):
        print(f"[{i+1}] BM25 Score: {res['score']:.4f}")
        print(f"    Chunk ID: {res['chunk'].id}")
        print(f"    Source: {res['chunk'].source_file} | Heading: {res['chunk'].section_heading}")
        print(f"    Text: {res['chunk'].text[:140]}...")
        print("-" * 50)

if __name__ == "__main__":
    main()

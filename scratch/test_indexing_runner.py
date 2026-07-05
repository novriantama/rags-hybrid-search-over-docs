import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent))

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import ChunkingOrchestrator, DocumentChunk
from src.retrieval.index_manager import IndexManager

def main():
    print("=== Hybrid RAG Ingestion & Indexing Integration Test ===")
    
    # 1. Load sample document
    loader = DocumentLoader()
    fragments = loader.load_processed_fragments()
    if not fragments:
        print("No processed fragments found. Run test_ingestion_runner.py first.")
        return
        
    print(f"Loaded {len(fragments)} fragments.")
    
    # 2. Chunk fragments recursively
    orchestrator = ChunkingOrchestrator()
    chunks = orchestrator.chunk_documents(fragments, "recursive")
    print(f"Generated {len(chunks)} chunks.")
    
    # 3. Initialize IndexManager & Reset database
    manager = IndexManager()
    print("Clearing database and sparse index files...")
    manager.clear_indexes()
    
    # 4. Index all chunks (the embedding generation will automatically fallback to dummy vectors in offline mode)
    print("\nIndexing chunks...")
    indexed, skipped = manager.index_chunks(chunks)
    print(f"Indexed: {indexed} | Skipped: {skipped}")
    
    # Verify DB counts
    print(f"ChromaDB collection count: {manager.dense_index.collection.count()}")
    print(f"BM25 corpus count: {len(manager.sparse_index.chunks)}")
    
    # 5. Attempt to insert a near-duplicate chunk
    print("\n--- Attempting Duplicate Ingestion ---")
    duplicate_chunk = DocumentChunk(
        id="duplicate_id_01",
        text=chunks[0].text, # Identical text to the first chunk
        source_file=chunks[0].source_file,
        section_heading=chunks[0].section_heading,
        page_number=chunks[0].page_number,
        file_type=chunks[0].file_type,
        chunking_strategy="recursive",
        character_count=chunks[0].character_count
    )
    
    indexed_dup, skipped_dup = manager.index_chunks([duplicate_chunk])
    print(f"Index Batch results for duplicate -> Indexed: {indexed_dup} | Skipped: {skipped_dup}")
    
    print(f"Final ChromaDB collection count: {manager.dense_index.collection.count()}")
    print(f"Final BM25 corpus count: {len(manager.sparse_index.chunks)}")
    
    # 6. Test Sparse keyword search
    print("\n--- Testing Sparse Keyword Search ---")
    query = "Postgres database"
    results = manager.sparse_index.search(query, k=2)
    print(f"Query: '{query}'")
    for i, res in enumerate(results):
        print(f"  Result {i+1}: Score={res['score']:.4f} | Text: {res['chunk'].text[:100]}...")

if __name__ == "__main__":
    main()

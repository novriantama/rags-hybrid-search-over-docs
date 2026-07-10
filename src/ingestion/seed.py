import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# pyrefly: ignore [missing-import]
from src.ingestion.loader import DocumentLoader
# pyrefly: ignore [missing-import]
from src.ingestion.chunker import ChunkingOrchestrator
# pyrefly: ignore [missing-import]
from src.retrieval.index_manager import IndexManager

def main():
    print("=== Seeding RAG Database Corpus ===")
    
    # Wait for ChromaDB to be responsive (retry up to 20 times)
    import time
    print("Waiting for database connection...")
    manager = None
    for attempt in range(20):
        try:
            temp_manager = IndexManager()
            temp_manager.dense_index.client.heartbeat()
            manager = temp_manager
            print("Database connection established!")
            break
        except Exception as e:
            if attempt == 19:
                print(f"Error: Could not connect to ChromaDB after 20 attempts: {e}")
                sys.exit(1)
            time.sleep(1)

    if manager is None:
        print("Error: IndexManager was not initialized. Exiting.")
        sys.exit(1)

    loader = DocumentLoader()
    orchestrator = ChunkingOrchestrator()
    
    # Clear previous indexes
    print("Clearing database and sparse indexes...")
    manager.clear_indexes()
    
    # Ingest raw markdown documents from raw storage folder
    raw_files = ["auth_service.md", "database_guide.md", "monitoring.md", "deployment_ops.md", "sample_doc.md"]
    all_chunks = []
    
    for filename in raw_files:
        path = loader.raw_dir / filename
        if path.exists():
            print(f"Ingesting: {filename}")
            fragments = loader.ingest_file(path)
            # Use recursive chunking strategy as default
            chunks = orchestrator.chunk_documents(fragments, "recursive")
            all_chunks.extend(chunks)
        else:
            print(f"File not found in raw workspace: {path}")
            
    if all_chunks:
        print(f"\nIndexing {len(all_chunks)} chunks...")
        indexed, skipped = manager.index_chunks(all_chunks)
        print(f"Successfully seeded: {indexed} chunks indexed ({skipped} skipped).")
    else:
        print("Warning: No chunks found to index.")

if __name__ == "__main__":
    main()

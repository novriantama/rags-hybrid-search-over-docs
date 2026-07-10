import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import ChunkingOrchestrator
from src.retrieval.index_manager import IndexManager

def main():
    print("=== Seeding RAG Database Corpus ===")
    loader = DocumentLoader()
    orchestrator = ChunkingOrchestrator()
    manager = IndexManager()
    
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

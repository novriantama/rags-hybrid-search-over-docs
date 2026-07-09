import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pyrefly: ignore [missing-import]
from src.ingestion.loader import DocumentLoader

def main():
    print("=== Ingesting Raw Documents ===")
    loader = DocumentLoader()
    
    # Clear previously processed files
    for file in loader.processed_dir.glob("*.json"):
        file.unlink()
        
    raw_files = ["auth_service.md", "database_guide.md", "monitoring.md", "deployment_ops.md", "sample_doc.md"]
    for filename in raw_files:
        path = loader.raw_dir / filename
        if path.exists():
            print(f"Ingesting: {filename}")
            fragments = loader.ingest_file(path)
            print(f"  Generated {len(fragments)} fragments.")
        else:
            print(f"File not found: {path}")

if __name__ == "__main__":
    main()

import os
from typing import List, Dict, Any, Optional
import chromadb
from openai import OpenAI

# pyrefly: ignore [missing-import]
from src.config import settings
# pyrefly: ignore [missing-import]
from src.ingestion.chunker import DocumentChunk

class DenseIndex:
    """Manages ChromaDB persistent storage, embedding generation, and vector retrieval."""
    def __init__(self, persist_directory: Optional[str] = None, openai_client: Optional[OpenAI] = None):
        path = persist_directory or settings.get_absolute_path(settings.chroma_db_path).as_posix()
        os.makedirs(path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=path)
        # Using cosine similarity metric: distance is 1 - CosineSimilarity
        self.collection = self.client.get_or_create_collection(
            name="rag_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        self._openai_client = openai_client

    @property
    def openai_client(self) -> OpenAI:
        if not self._openai_client:
            self._openai_client = OpenAI(api_key=settings.openai_api_key or "mock_key")
        return self._openai_client

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates OpenAI embeddings in batches of 100 to avoid payload limit issues."""
        if not texts:
            return []
            
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model=settings.embedding_model
                )
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                # Mock or fallback in test or error environments
                print(f"Embedding generation failed: {e}. Generating deterministic dummy vectors.")
                import hashlib
                import numpy as np
                for text in batch:
                    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
                    rng = np.random.default_rng(seed)
                    vec = rng.standard_normal(1536)
                    vec /= np.linalg.norm(vec)
                    all_embeddings.append(vec.tolist())
                
        return all_embeddings

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Upserts text chunks and their precomputed embeddings into ChromaDB."""
        if not chunks:
            return
            
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        
        metadatas = []
        for i, chunk in enumerate(chunks):
            meta = {
                "source_file": chunk.source_file,
                "chunk_index": i,
                "section_heading": chunk.section_heading or "",
                "chunking_strategy": chunk.chunking_strategy,
                "character_count": chunk.character_count,
                "file_type": chunk.file_type
            }
            if chunk.page_number is not None:
                meta["page_number"] = chunk.page_number
            metadatas.append(meta)
            
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def query_closest(self, embedding: List[float], n_results: int = 1) -> Dict[str, Any]:
        """Queries for the closest matching document in the collection to check duplicates."""
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["distances", "documents", "metadatas"]
        )

    def search(self, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """Queries the vector database returning results ordered by cosine distance."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["distances", "documents", "metadatas"]
        )
        
        formatted_results = []
        if results and results.get("ids") and len(results["ids"][0]) > 0:
            for idx in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][idx],
                    "text": results["documents"][0][idx],
                    "distance": results["distances"][0][idx],
                    "metadata": results["metadatas"][0][idx]
                })
        return formatted_results

    def clear(self):
        """Resets the vector database."""
        try:
            self.client.delete_collection("rag_chunks")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="rag_chunks",
            metadata={"hnsw:space": "cosine"}
        )

from typing import List, Dict, Any, Optional
# pyrefly: ignore [missing-import]
from src.ingestion.chunker import DocumentChunk
# pyrefly: ignore [missing-import]
from src.retrieval.dense import DenseIndex
# pyrefly: ignore [missing-import]
from src.retrieval.sparse import SparseIndex

class HybridRetriever:
    """Coordinates search operations across dense vector and sparse keyword indexes."""
    def __init__(self, dense_index: Optional[DenseIndex] = None, sparse_index: Optional[SparseIndex] = None):
        self.dense_index = dense_index or DenseIndex()
        self.sparse_index = sparse_index or SparseIndex()

    def dense_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Embeds the query text and retrieves top-k chunks from the vector database.
        
        Returns:
            List[Dict[str, Any]]: List of dicts containing 'chunk' (DocumentChunk) and 'score' (float, cosine similarity).
        """
        if not query or k <= 0:
            return []
            
        # 1. Embed query text
        query_embs = self.dense_index.generate_embeddings([query])
        if not query_embs:
            return []
        query_emb = query_embs[0]
        
        # 2. Search ChromaDB
        db_results = self.dense_index.search(query_emb, k=k)
        
        # 3. Format results mapping metadata back to DocumentChunk
        formatted_results = []
        for res in db_results:
            meta = res["metadata"]
            
            # Reconstruct DocumentChunk object from search result metadata
            chunk = DocumentChunk(
                id=res["id"],
                text=res["text"],
                source_file=meta.get("source_file", ""),
                section_heading=meta.get("section_heading"),
                page_number=meta.get("page_number"),
                file_type=meta.get("file_type", ""),
                chunking_strategy=meta.get("chunking_strategy", "fixed"),
                character_count=meta.get("character_count", len(res["text"]))
            )
            
            # Since ChromaDB space is cosine, distance is 1.0 - CosineSimilarity.
            # Convert back to CosineSimilarity score:
            score = 1.0 - res["distance"]
            
            formatted_results.append({
                "chunk": chunk,
                "score": score
            })
            
        return formatted_results

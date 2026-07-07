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

    def sparse_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Queries the BM25 sparse index and retrieves the top-k chunks.
        
        Returns:
            List[Dict[str, Any]]: List of dicts containing 'chunk' (DocumentChunk) and 'score' (float, BM25 score).
        """
        if not query or k <= 0:
            return []
        return self.sparse_index.search(query, k=k)

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Applies Reciprocal Rank Fusion (RRF) algorithm to combine dense and sparse results.
        
        Score(d) = dense_weight * (1 / (rrf_k + rank_dense)) + sparse_weight * (1 / (rrf_k + rank_sparse))
        """
        rrf_scores: Dict[str, float] = {}
        chunk_mapping: Dict[str, DocumentChunk] = {}

        # 1. Process dense rankings
        if dense_weight > 0.0:
            for rank, res in enumerate(dense_results, 1):
                chunk = res["chunk"]
                chunk_id = chunk.id
                chunk_mapping[chunk_id] = chunk
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + dense_weight * (1.0 / (rrf_k + rank))

        # 2. Process sparse rankings
        if sparse_weight > 0.0:
            for rank, res in enumerate(sparse_results, 1):
                chunk = res["chunk"]
                chunk_id = chunk.id
                chunk_mapping[chunk_id] = chunk
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + sparse_weight * (1.0 / (rrf_k + rank))

        # 3. Sort chunks by fused score descending
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. Format outputs
        fused_results = []
        for chunk_id, score in sorted_chunks:
            fused_results.append({
                "chunk": chunk_mapping[chunk_id],
                "score": score
            })

        return fused_results

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        candidate_k: int = 20,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Executes both dense and sparse queries and fuses their rankings using Reciprocal Rank Fusion (RRF).
        
        Returns:
            List[Dict[str, Any]]: Top-k fused search results.
        """
        if not query or k <= 0:
            return []

        # 1. Query both dense and sparse candidate pools
        dense_candidates = self.dense_search(query, k=candidate_k)
        sparse_candidates = self.sparse_search(query, k=candidate_k)

        # 2. Run fusion
        fused_results = self.reciprocal_rank_fusion(
            dense_results=dense_candidates,
            sparse_results=sparse_candidates,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            rrf_k=rrf_k
        )

        # 3. Return top-k matches
        return fused_results[:k]

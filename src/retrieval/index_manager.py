import logging
from typing import List, Tuple, Optional
import numpy as np
from openai import OpenAI

from src.ingestion.chunker import DocumentChunk
from src.retrieval.dense import DenseIndex
from src.retrieval.sparse import SparseIndex

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class IndexManager:
    """Orchestrates dense (ChromaDB) and sparse (BM25) search indexes and enforces near-duplicate filters."""
    def __init__(
        self,
        dense_index: Optional[DenseIndex] = None,
        sparse_index: Optional[SparseIndex] = None,
        openai_client: Optional[OpenAI] = None
    ):
        self.dense_index = dense_index or DenseIndex(openai_client=openai_client)
        self.sparse_index = sparse_index or SparseIndex()

    def index_chunks(self, chunks: List[DocumentChunk]) -> Tuple[int, int]:
        """
        Embeds, deduplicates, and stores text chunks.
        
        Checks for duplicates (cosine similarity > 0.95) against both:
        1. Existing stored chunks in the vector database.
        2. Other chunks inside the same indexing batch.
        
        Returns:
            Tuple[int, int]: (number of chunks indexed, number of chunks skipped as duplicates)
        """
        if not chunks:
            return 0, 0
            
        # 1. Precompute embeddings in batch to optimize API calls
        texts = [chunk.text for chunk in chunks]
        embeddings = self.dense_index.generate_embeddings(texts)
        
        indexed_chunks: List[DocumentChunk] = []
        indexed_embeddings: List[List[float]] = []
        skipped_count = 0
        
        for chunk, embedding in zip(chunks, embeddings):
            is_duplicate = False
            
            # A. Check similarity against chunks already persisted in ChromaDB
            if self.dense_index.collection.count() > 0:
                closest = self.dense_index.query_closest(embedding, n_results=1)
                if closest and closest.get("distances") and len(closest["distances"][0]) > 0:
                    distance = closest["distances"][0][0]
                    similarity = 1.0 - distance
                    if similarity > 0.95:
                         logger.info(
                             f"Flagged duplicate against DB (similarity: {similarity:.4f} > 0.95). "
                             f"Skipping chunk: {chunk.id}"
                         )
                         is_duplicate = True
                         skipped_count += 1
                         continue
                         
            # B. Check similarity against other chunks in the same upload batch
            for prev_emb in indexed_embeddings:
                norm_a = np.linalg.norm(embedding)
                norm_b = np.linalg.norm(prev_emb)
                if norm_a > 0 and norm_b > 0:
                    sim = float(np.dot(embedding, prev_emb) / (norm_a * norm_b))
                    if sim > 0.95:
                        logger.info(
                            f"Flagged duplicate in current batch (similarity: {sim:.4f} > 0.95). "
                            f"Skipping chunk: {chunk.id}"
                        )
                        is_duplicate = True
                        skipped_count += 1
                        break
                        
            if not is_duplicate:
                indexed_chunks.append(chunk)
                indexed_embeddings.append(embedding)
                
        # 3. Synchronously commit changes to both indexes
        if indexed_chunks:
            self.dense_index.add_chunks(indexed_chunks, indexed_embeddings)
            self.sparse_index.add_chunks(indexed_chunks)
            
        return len(indexed_chunks), skipped_count

    def clear_indexes(self):
        """Clears both dense and sparse indexes."""
        self.dense_index.clear()
        self.sparse_index.clear()

import os
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingestion.chunker import DocumentChunk

class SparseIndex:
    """Manages BM25 keyword matching corpus, dynamic re-indexing, and persistence."""
    def __init__(self, persist_path: Optional[str] = None):
        if persist_path:
            self.persist_path = Path(persist_path)
        else:
            data_root = settings.get_absolute_path(settings.data_dir)
            self.persist_path = data_root.parent / "bm25" / "bm25_index.pkl"
            
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        
        self.load()

    def tokenize(self, text: str) -> List[str]:
        """Simple alphanumeric tokenizer that lowercases terms."""
        return re.findall(r'\w+', text.lower())

    def add_chunks(self, new_chunks: List[DocumentChunk]):
        """Adds new chunks to the corpus, fits a fresh BM25 index, and persists it."""
        if not new_chunks:
            return
            
        self.chunks.extend(new_chunks)
        self._rebuild_index()
        self.save()

    def _rebuild_index(self):
        """Builds a new BM25Okapi search model from raw chunk texts."""
        if not self.chunks:
            self.bm25 = None
            return
            
        tokenized_corpus = [self.tokenize(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Queries the corpus using BM25 token frequencies, returning chunks with scores."""
        if not self.bm25 or not self.chunks:
            return []
            
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get indices of sorted scores descending
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            # Only return chunks that have some relevance (score > 0)
            if score > 0.0:
                results.append({
                    "chunk": self.chunks[idx],
                    "score": score
                })
        return results

    def save(self):
        """Serializes current chunks list to file directory."""
        os.makedirs(self.persist_path.parent, exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self):
        """Deserializes chunks and fits BM25 index if pickle exists."""
        if not self.persist_path.exists():
            self.chunks = []
            self.bm25 = None
            return
            
        try:
            with open(self.persist_path, "rb") as f:
                self.chunks = pickle.load(f)
            self._rebuild_index()
        except Exception as e:
            print(f"Error loading sparse index pickle ({e}). Starting fresh.")
            self.chunks = []
            self.bm25 = None

    def clear(self):
        """Cleans the cached index file."""
        if self.persist_path.exists():
            try:
                self.persist_path.unlink()
            except Exception:
                pass
        self.chunks = []
        self.bm25 = None

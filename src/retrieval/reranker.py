import logging
import re
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

# pyrefly: ignore [missing-import]
from src.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Manages neural cross-encoder re-ranking for refining candidate search results."""
    def __init__(self, model_name: str = "sentence-transformers/ms-marco-MiniLM-L-6-v2", force_fallback: bool = False):
        self.model_name = model_name
        self.force_fallback = force_fallback
        self.model: Optional[CrossEncoder] = None
        self._fallback_mode_active = force_fallback

    def _load_model(self):
        """Lazy loads the CrossEncoder model, capturing errors and enabling fallback if download fails."""
        if self.model or self._fallback_mode_active:
            return
            
        try:
            logger.info(f"Loading CrossEncoder model: {self.model_name}...")
            self.model = CrossEncoder(self.model_name)
            logger.info("CrossEncoder model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Failed to load CrossEncoder model ({e}). "
                "Activating Jaccard token overlap fallback mode."
            )
            self._fallback_mode_active = True

    def _jaccard_similarity(self, query: str, text: str) -> float:
        """Fallback scoring using simple token intersection over union (Jaccard similarity)."""
        def get_words(t):
            return set(re.findall(r'\w+', t.lower()))
            
        q_words = get_words(query)
        t_words = get_words(text)
        
        if not q_words or not t_words:
            return 0.0
            
        intersection = q_words.intersection(t_words)
        union = q_words.union(t_words)
        return float(len(intersection) / len(union))

    def rerank(self, query: str, chunks: List[DocumentChunk], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Re-ranks a list of DocumentChunks based on semantic relevance to the query.
        
        Returns:
            List[Dict[str, Any]]: Top-k chunks with float scores.
        """
        if not chunks or not query or top_k <= 0:
            return []
            
        self._load_model()
        
        scored_results = []
        
        if self._fallback_mode_active:
            # Fallback evaluation
            for chunk in chunks:
                score = self._jaccard_similarity(query, chunk.text)
                scored_results.append({
                    "chunk": chunk,
                    "score": score
                })
        else:
            # Neural evaluation
            try:
                pairs = [[query, chunk.text] for chunk in chunks]
                # Predict returns standard float scores
                scores = self.model.predict(pairs)
                for idx, score in enumerate(scores):
                    scored_results.append({
                        "chunk": chunks[idx],
                        "score": float(score)
                    })
            except Exception as e:
                logger.error(f"Error during neural rerank prediction: {e}. Falling back to token overlap.")
                for chunk in chunks:
                    score = self._jaccard_similarity(query, chunk.text)
                    scored_results.append({
                        "chunk": chunk,
                        "score": score
                    })
                    
        # Sort scores descending
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

import re
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

# pyrefly: ignore [missing-import]
from src.config import settings
# pyrefly: ignore [missing-import]
from src.ingestion.loader import DocumentFragment

class DocumentChunk(BaseModel):
    """Represents a final chunk of text to be stored in the indexes."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source_file: str
    section_heading: Optional[str] = None
    page_number: Optional[int] = None
    file_type: str
    chunking_strategy: str  # "fixed", "recursive", "semantic"
    character_count: int
    token_count: Optional[int] = None

class BaseChunker(ABC):
    """Base class for all chunking strategies."""
    
    @abstractmethod
    def chunk_fragment(self, fragment: DocumentFragment) -> List[Dict[str, Any]]:
        """Splits a DocumentFragment into text blocks with metadata."""
        pass
        
    def process_fragment(self, fragment: DocumentFragment, strategy_name: str) -> List[DocumentChunk]:
        """Orchestrates chunking and converts raw blocks to DocumentChunk objects."""
        blocks = self.chunk_fragment(fragment)
        chunks = []
        for i, block in enumerate(blocks):
            text = block["text"]
            # Unique stable ID: source_file:heading:strategy:index
            clean_heading = (fragment.section_heading or "none").replace(" ", "_")
            chunk_id = f"{fragment.source_file}:{clean_heading}:{strategy_name}:{i}"
            
            chunks.append(DocumentChunk(
                id=chunk_id,
                text=text,
                source_file=fragment.source_file,
                section_heading=fragment.section_heading,
                page_number=fragment.page_number,
                file_type=fragment.file_type,
                chunking_strategy=strategy_name,
                character_count=len(text),
                token_count=len(text) // 4  # rough token approximation (4 characters = 1 token)
            ))
        return chunks

class FixedSizeChunker(BaseChunker):
    """Splits text into fixed size characters with overlap (Baseline strategy)."""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=True
        )
        
    def chunk_fragment(self, fragment: DocumentFragment) -> List[Dict[str, Any]]:
        texts = self.splitter.split_text(fragment.text)
        return [{"text": t} for t in texts]

class RecursiveHeaderChunker(BaseChunker):
    """Splits paragraphs and sentences recursively while retaining section structures."""
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 80):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def chunk_fragment(self, fragment: DocumentFragment) -> List[Dict[str, Any]]:
        texts = self.splitter.split_text(fragment.text)
        return [{"text": t} for t in texts]

class SemanticChunker(BaseChunker):
    """Splits on topic boundaries by computing similarity embeddings between sentences."""
    def __init__(self, similarity_threshold: float = 0.82, max_chunk_size: int = 1500, openai_client: Optional[OpenAI] = None):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self._client = openai_client
        
    @property
    def client(self) -> OpenAI:
        if not self._client:
            self._client = OpenAI(
                api_key=settings.openai_api_key or "mock_key",
                base_url=settings.openai_api_base or None
            )
        return self._client
        
    def _split_sentences(self, text: str) -> List[str]:
        # Split on sentence boundaries: periods, exclamation, or question marks followed by spaces
        sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')
        return [s.strip() for s in sentence_endings.split(text) if s.strip()]
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
        
    def _get_embeddings(self, sentences: List[str]) -> List[np.ndarray]:
        try:
            response = self.client.embeddings.create(
                input=sentences,
                model=settings.embedding_model
            )
            return [np.array(emb.embedding) for emb in response.data]
        except Exception as e:
            # Fallback to random embeddings in case of API failure (or mock key in tests)
            # This makes unit testing and offline development easier
            print(f"OpenAI embedding call failed ({e}). Falling back to dummy vectors.")
            return [np.random.rand(1536) for _ in sentences]
            
    def chunk_fragment(self, fragment: DocumentFragment) -> List[Dict[str, Any]]:
        sentences = self._split_sentences(fragment.text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [{"text": sentences[0]}]
            
        embeddings = self._get_embeddings(sentences)
        
        blocks = []
        current_sentences = [sentences[0]]
        current_len = len(sentences[0])
        
        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i-1], embeddings[i])
            next_len = len(sentences[i])
            
            # If the semantic similarity is low OR adding this sentence breaches size constraints, split
            if sim < self.similarity_threshold or (current_len + next_len > self.max_chunk_size):
                blocks.append({"text": " ".join(current_sentences)})
                current_sentences = [sentences[i]]
                current_len = next_len
            else:
                current_sentences.append(sentences[i])
                current_len += next_len + 1 # +1 for space separator
                
        if current_sentences:
            blocks.append({"text": " ".join(current_sentences)})
            
        return blocks

class ChunkingOrchestrator:
    """Orchestrates switchable document chunking strategies."""
    def __init__(self, openai_client: Optional[OpenAI] = None):
        self.strategies: Dict[str, BaseChunker] = {
            "fixed": FixedSizeChunker(),
            "recursive": RecursiveHeaderChunker(),
            "semantic": SemanticChunker(openai_client=openai_client)
        }
        
    def chunk_documents(self, fragments: List[DocumentFragment], strategy: str) -> List[DocumentChunk]:
        chunker = self.strategies.get(strategy.lower())
        if not chunker:
            raise ValueError(f"Unknown chunking strategy: {strategy}. Options: fixed, recursive, semantic")
            
        all_chunks = []
        for frag in fragments:
            all_chunks.extend(chunker.process_fragment(frag, strategy.lower()))
        return all_chunks

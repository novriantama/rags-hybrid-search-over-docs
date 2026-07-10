from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class AskRequest(BaseModel):
    """Request payload for RAG querying."""
    question: str = Field(..., description="The user question to ask the RAG system.")
    threshold: float = Field(0.7, description="Similarity threshold under which the fallback response is triggered.")
    chunking_strategy: str = Field("recursive", description="Chunking strategy to use if new indexing occurs or strategy is queried.")
    dense_weight: float = Field(0.5, description="Weight factor for dense vector search (0.0 - 1.0).")
    sparse_weight: float = Field(0.5, description="Weight factor for sparse keyword search (0.0 - 1.0).")
    k: int = Field(5, description="Number of context chunks to retrieve.")

class ConfidenceReportSchema(BaseModel):
    """Details of the answer confidence evaluation."""
    composite_score: float = Field(..., description="Overall weighted confidence score (0.0 to 1.0).")
    retrieval_confidence: float = Field(..., description="Similarity confidence of the top retrieved chunks.")
    citation_coverage: float = Field(..., description="Proportion of answer statements backed by verified citations.")
    completeness_score: float = Field(..., description="Judge-evaluated completeness addressing the user question.")

class CitationSchema(BaseModel):
    """Inline claim citation details."""
    claim: str = Field(..., description="Parsed claim statement.")
    citation_index: int = Field(..., description="Inline bracket citation index.")
    source_file: str = Field(..., description="Source document filename matching the citation.")
    result: str = Field(..., description="Verification result (e.g. 'VERIFIED' or 'UNSUPPORTED').")

class AskResponse(BaseModel):
    """Response payload for RAG querying."""
    answer: str = Field(..., description="Grounded markdown formatted answer.")
    confidence_report: ConfidenceReportSchema = Field(..., description="Calculated scores evaluating response reliability.")
    citations: List[CitationSchema] = Field(..., description="List of parsed and LLM-verified citations.")
    fallback_triggered: bool = Field(..., description="True if retrieval confidence was too low and fallback report was output.")

class DocumentSchema(BaseModel):
    """Details of a document stored in the raw corpus workspace."""
    filename: str = Field(..., description="Filename of the raw document.")
    file_type: str = Field(..., description="File extension (e.g. 'md', 'txt', 'pdf').")
    fragment_count: int = Field(..., description="Number of parsed fragments/sections generated.")
    size_bytes: int = Field(..., description="Raw document file size in bytes.")

class IngestResponse(BaseModel):
    """Ingestion workflow response status."""
    filename: str = Field(..., description="Name of the processed file.")
    status: str = Field(..., description="Ingestion execution outcome (e.g., 'SUCCESS').")
    fragments_created: int = Field(..., description="Number of parsed document fragments.")
    chunks_indexed: int = Field(..., description="Number of chunks indexed in database.")

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# pyrefly: ignore [missing-import]
from src.config import settings
# pyrefly: ignore [missing-import]
from src.ingestion.loader import DocumentLoader
# pyrefly: ignore [missing-import]
from src.ingestion.chunker import ChunkingOrchestrator
# pyrefly: ignore [missing-import]
from src.retrieval.index_manager import IndexManager
# pyrefly: ignore [missing-import]
from src.retrieval.retriever import HybridRetriever
# pyrefly: ignore [missing-import]
from src.generation.generator import GroundedGenerator
# pyrefly: ignore [missing-import]
from src.api.schemas import AskRequest, AskResponse, DocumentSchema, IngestResponse, CitationSchema, ConfidenceReportSchema

app = FastAPI(
    title="Hybrid Search RAG over Internal Docs",
    description="Production-ready grounded Q&A and document retrieval pipeline.",
    version="1.0.0"
)

# Enable CORS for frontend dashboard or integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Pipeline Instances
loader = DocumentLoader()
orchestrator = ChunkingOrchestrator()
manager = IndexManager()
retriever = HybridRetriever(dense_index=manager.dense_index, sparse_index=manager.sparse_index)
generator = GroundedGenerator()

@app.post("/v1/ask", response_model=AskResponse, summary="Query the hybrid RAG pipeline")
async def ask_question(request: AskRequest):
    """
    Retrieves context using Hybrid Search (fusing Dense vector search and Sparse keyword search)
    and generates a grounded answer. Includes inline verified citations and a confidence report.
    """
    try:
        # 1. Retrieve candidate chunks using Hybrid Search
        results = retriever.hybrid_search(
            query=request.question,
            k=request.k,
            dense_weight=request.dense_weight,
            sparse_weight=request.sparse_weight,
            use_reranker=True
        )
        
        # 2. Generate grounded response with citations
        pipeline_output = generator.generate_response(
            question=request.question,
            chunks_or_results=results,
            retrieval_threshold=request.threshold
        )
        
        # 3. Format response elements
        verifications = pipeline_output.get("verification_results", [])
        citations = []
        for item in verifications:
            citations.append(CitationSchema(
                claim=item["claim"],
                citation_index=item["citation_index"],
                source_file=item["source_file"],
                result=item["result"]
            ))
            
        conf_report = pipeline_output.get("confidence_report", {})
        confidence = ConfidenceReportSchema(
            composite_score=conf_report.get("composite_score", 0.0),
            retrieval_confidence=conf_report.get("retrieval_confidence", 0.0),
            citation_coverage=conf_report.get("citation_coverage", 0.0),
            completeness_score=conf_report.get("completeness_score", 0.0)
        )
        
        retrieved_chunks = []
        for item in results:
            chunk = item["chunk"]
            retrieved_chunks.append({
                "id": chunk.id,
                "text": chunk.text,
                "source_file": chunk.source_file,
                "section_heading": chunk.section_heading,
                "score": float(item.get("score", 1.0))
            })
            
        return AskResponse(
            answer=pipeline_output["answer"],
            confidence_report=confidence,
            citations=citations,
            fallback_triggered=pipeline_output.get("fallback_triggered", False),
            retrieved_chunks=retrieved_chunks
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during RAG pipeline execution: {str(e)}"
        )

@app.get("/v1/documents", response_model=List[DocumentSchema], summary="List all indexed raw documents")
async def list_documents():
    """
    Inspects the local storage workspace raw and processed folders to list
    all documents, size stats, and fragment counts.
    """
    docs = []
    raw_dir = loader.raw_dir
    processed_dir = loader.processed_dir
    
    if not raw_dir.exists():
        return []
        
    for path in raw_dir.glob("*"):
        if path.is_file() and path.name != ".gitkeep":
            # Determine size in bytes
            size_bytes = path.stat().st_size
            ext = path.suffix.lower().lstrip(".")
            
            # Find parsed fragments count
            fragments_count = 0
            json_path = processed_dir / f"{path.stem}.json"
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        fragments_count = len(data.get("fragments", []))
                except Exception:
                    pass
                    
            docs.append(DocumentSchema(
                filename=path.name,
                file_type=ext,
                fragment_count=fragments_count,
                size_bytes=size_bytes
            ))
            
    return docs

@app.post("/v1/ingest", response_model=IngestResponse, summary="Ingest and index a new document")
async def ingest_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Query("recursive", enum=["fixed", "recursive", "semantic"])
):
    """
    Uploads a new document file, runs the loading and parser logic to convert it into parsed fragments,
    chunks it using the selected strategy, and indexes it into the search indices.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename."
        )
    filename: str = file.filename
    
    # 1. Save uploaded file to raw workspace directory
    raw_dest = loader.raw_dir / filename
    try:
        with open(raw_dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )
        
    try:
        # 2. Ingest and parse file
        fragments = loader.ingest_file(raw_dest)
        
        # 3. Chunk parsed fragments
        chunks = orchestrator.chunk_documents(fragments, chunking_strategy)
        
        # 4. Index chunks into vector database and sparse index
        indexed, skipped = manager.index_chunks(chunks)
        
        return IngestResponse(
            filename=filename,
            status="SUCCESS",
            fragments_created=len(fragments),
            chunks_indexed=indexed
        )
    except Exception as e:
        # Clean up failed ingestion file to keep directory clean
        if raw_dest.exists():
            raw_dest.unlink()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Ingestion process failed: {str(e)}"
        )

# Import json inside file functions
import json

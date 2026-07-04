import pytest
from unittest.mock import MagicMock
# pyrefly: ignore [missing-import]
from src.ingestion.loader import DocumentFragment
# pyrefly: ignore [missing-import]
from src.ingestion.chunker import (
    FixedSizeChunker,
    RecursiveHeaderChunker,
    SemanticChunker,
    ChunkingOrchestrator,
)

@pytest.fixture
def sample_fragment():
    return DocumentFragment(
        text="Line one of document text. Line two of document text. Line three which is longer and goes onto detail database parameters.",
        source_file="guide.md",
        section_heading="Guidelines",
        page_number=1,
        file_type="md"
    )

def test_fixed_size_chunker(sample_fragment):
    # Split text into small chunks of size 30 with 5 overlap
    chunker = FixedSizeChunker(chunk_size=30, chunk_overlap=5)
    chunks = chunker.process_fragment(sample_fragment, "fixed")
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.chunking_strategy == "fixed"
        assert chunk.source_file == "guide.md"
        assert chunk.section_heading == "Guidelines"
        assert len(chunk.text) <= 30

def test_recursive_header_chunker(sample_fragment):
    # Splits recursively, paragraph > sentence > word
    chunker = RecursiveHeaderChunker(chunk_size=40, chunk_overlap=10)
    chunks = chunker.process_fragment(sample_fragment, "recursive")
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.chunking_strategy == "recursive"
        assert chunk.page_number == 1
        assert len(chunk.text) <= 40

def test_semantic_chunker_logic():
    # Construct mock OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    
    # We will simulate 3 sentences
    emb1 = MagicMock()
    emb1.embedding = [1.0, 0.0, 0.0]
    emb2 = MagicMock()
    emb2.embedding = [0.98, 0.01, 0.0] # High similarity with emb1
    emb3 = MagicMock()
    emb3.embedding = [0.0, 1.0, 0.0] # Low similarity with emb2 (perpendicular)
    
    mock_response.data = [emb1, emb2, emb3]
    mock_client.embeddings.create.return_value = mock_response
    
    chunker = SemanticChunker(similarity_threshold=0.8, max_chunk_size=1000, openai_client=mock_client)
    
    frag = DocumentFragment(
        text="This is sentence one. This is sentence two. This is sentence three.",
        source_file="test.txt",
        section_heading="Mock",
        page_number=1,
        file_type="txt"
    )
    
    chunks = chunker.process_fragment(frag, "semantic")
    
    # Sentence 1 & 2 similarity is high (~0.98 >= 0.8) -> Grouped
    # Sentence 2 & 3 similarity is 0.0 (< 0.8) -> Split
    # Result: 2 chunks
    assert len(chunks) == 2
    assert chunks[0].text == "This is sentence one. This is sentence two."
    assert chunks[1].text == "This is sentence three."
    assert chunks[0].chunking_strategy == "semantic"

def test_chunking_orchestrator(sample_fragment):
    orchestrator = ChunkingOrchestrator()
    
    # Should resolve fixed strategy
    chunks = orchestrator.chunk_documents([sample_fragment], "fixed")
    assert len(chunks) > 0
    assert chunks[0].chunking_strategy == "fixed"
    
    # Should resolve recursive strategy
    chunks = orchestrator.chunk_documents([sample_fragment], "recursive")
    assert len(chunks) > 0
    assert chunks[0].chunking_strategy == "recursive"
    
    # Should throw error for unknown strategy
    with pytest.raises(ValueError):
        orchestrator.chunk_documents([sample_fragment], "invalid_strategy")

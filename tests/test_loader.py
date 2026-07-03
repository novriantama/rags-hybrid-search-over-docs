import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.ingestion.loader import (
    DocumentFragment,
    TextParser,
    MarkdownParser,
    HTMLParser,
    PDFParser,
    DocumentLoader,
)

def test_text_parser(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello World!\nThis is a simple text file.", encoding="utf-8")
    
    parser = TextParser()
    fragments = parser.parse(txt_file)
    
    assert len(fragments) == 1
    assert fragments[0]["text"] == "Hello World!\nThis is a simple text file."
    assert fragments[0]["section_heading"] == "Document Start"
    assert fragments[0]["page_number"] == 1

def test_markdown_parser(tmp_path):
    md_file = tmp_path / "test.md"
    md_content = """# Main Title
Introductory text here.

## Section 1
Content of section 1.

### Subsection 1.1
Content of subsection 1.1.
"""
    md_file.write_text(md_content, encoding="utf-8")
    
    parser = MarkdownParser()
    fragments = parser.parse(md_file)
    
    assert len(fragments) == 3
    
    # Fragment 1: Introduction (before any section or right under # Main Title)
    assert fragments[0]["section_heading"] == "Main Title"
    assert "Introductory text here." in fragments[0]["text"]
    
    # Fragment 2: Section 1
    assert fragments[1]["section_heading"] == "Section 1"
    assert "Content of section 1." in fragments[1]["text"]
    
    # Fragment 3: Subsection 1.1
    assert fragments[2]["section_heading"] == "Subsection 1.1"
    assert "Content of subsection 1.1." in fragments[2]["text"]

def test_html_parser(tmp_path):
    html_file = tmp_path / "test.html"
    html_content = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Heading 1</h1>
            <p>First paragraph text. <span>Nested span text.</span></p>
            <h2>Heading 2</h2>
            <p>Second paragraph text.</p>
        </body>
    </html>
    """
    html_file.write_text(html_content, encoding="utf-8")
    
    parser = HTMLParser()
    fragments = parser.parse(html_file)
    
    assert len(fragments) == 2
    
    assert fragments[0]["section_heading"] == "Heading 1"
    assert "First paragraph text." in fragments[0]["text"]
    assert "Nested span text." in fragments[0]["text"]
    
    assert fragments[1]["section_heading"] == "Heading 2"
    assert "Second paragraph text." in fragments[1]["text"]

@patch("src.ingestion.loader.PdfReader")
def test_pdf_parser(mock_pdf_reader, tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("Dummy content", encoding="utf-8") # needs to exist
    
    # Set up mocks for PdfReader
    page1 = MagicMock()
    page1.extract_text.return_value = "Short Title Heading\nThis is page 1 body."
    
    page2 = MagicMock()
    page2.extract_text.return_value = "This is page 2 body. The first line is really long and should not be used as a header."
    
    mock_reader = MagicMock()
    mock_reader.pages = [page1, page2]
    mock_pdf_reader.return_value = mock_reader
    
    parser = PDFParser()
    fragments = parser.parse(pdf_file)
    
    assert len(fragments) == 2
    assert fragments[0]["page_number"] == 1
    assert fragments[0]["section_heading"] == "Short Title Heading"
    assert "This is page 1 body." in fragments[0]["text"]
    
    assert fragments[1]["page_number"] == 2
    assert fragments[1]["section_heading"] == "Page 2"
    assert "This is page 2 body." in fragments[1]["text"]

def test_document_loader_ingest_and_load(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    
    loader = DocumentLoader(raw_dir=raw_dir, processed_dir=processed_dir)
    
    # Create test markdown file
    test_file = tmp_path / "sample.md"
    test_file.write_text("# Welcome\nHello from markdown.", encoding="utf-8")
    
    # Ingest the file
    fragments = loader.ingest_file(test_file)
    
    assert len(fragments) == 1
    assert fragments[0].source_file == "sample.md"
    assert fragments[0].section_heading == "Welcome"
    assert fragments[0].file_type == "md"
    
    # Check that raw file was copied
    assert (raw_dir / "sample.md").exists()
    
    # Check that processed JSON was created
    processed_json = processed_dir / "sample.json"
    assert processed_json.exists()
    
    # Verify JSON content matches
    with open(processed_json, "r") as f:
        data = json.load(f)
        assert data["source_file"] == "sample.md"
        assert len(data["fragments"]) == 1
        assert data["fragments"][0]["section_heading"] == "Welcome"
        
    # Test loading processed fragments
    loaded = loader.load_processed_fragments()
    assert len(loaded) == 1
    assert loaded[0].source_file == "sample.md"
    assert loaded[0].section_heading == "Welcome"

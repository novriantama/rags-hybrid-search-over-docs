import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from pypdf import PdfReader

# pyrefly: ignore [missing-import]
from src.config import settings

class DocumentFragment(BaseModel):
    """A normalized chunk/fragment of a document with source metadata."""
    text: str
    source_file: str
    section_heading: Optional[str] = None
    page_number: Optional[int] = None
    file_type: str

class BaseParser:
    """Interface for document parsers."""
    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        raise NotImplementedError("Parsers must implement parse()")

class TextParser(BaseParser):
    """Parses plain text files."""
    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        if not content.strip():
            return []
            
        return [{
            "text": content.strip(),
            "section_heading": "Document Start",
            "page_number": 1
        }]

class MarkdownParser(BaseParser):
    """Parses markdown files, splitting text logically by headings."""
    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        fragments = []
        current_heading = "Introduction"
        current_lines: List[str] = []
        
        for line in lines:
            stripped = line.strip()
            # Detect markdown headers (e.g., # Header, ## Subheader)
            if stripped.startswith("#"):
                # Flush previous section
                text_content = "".join(current_lines).strip()
                if text_content:
                    fragments.append({
                        "text": text_content,
                        "section_heading": current_heading,
                        "page_number": 1
                    })
                    current_lines = []
                # Extract new header name, removing the # symbols and stripping spaces
                header_parts = stripped.split("#", 1)
                header_text = header_parts[-1].strip() if len(header_parts) > 1 else stripped
                # Clean up leading/trailing characters
                current_heading = header_text.lstrip("#").strip()
            else:
                current_lines.append(line)
                
        # Flush the remaining text at the end of the file
        text_content = "".join(current_lines).strip()
        if text_content:
            fragments.append({
                "text": text_content,
                "section_heading": current_heading,
                "page_number": 1
            })
            
        return fragments

class HTMLParser(BaseParser):
    """Parses HTML files, traversing structural tags and keeping heading association."""
    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
            
        soup = BeautifulSoup(html_content, "html.parser")
        # Remove script and style blocks
        for tag in soup(["script", "style"]):
            tag.decompose()
            
        body = soup.body if soup.body else soup
        
        state = {
            "heading": "Introduction",
            "fragments": [],
            "buffer": []
        }
        
        def walk(element):
            # If element is a header, flush current buffer and update active heading
            if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                if state["buffer"]:
                    text = "\n".join(state["buffer"]).strip()
                    if text:
                        state["fragments"].append({
                            "text": text,
                            "section_heading": state["heading"],
                            "page_number": 1
                        })
                    state["buffer"] = []
                state["heading"] = element.get_text(strip=True)
                return
                
            # If it's a text-bearing leaf element
            if not element.find_all(True) and element.string:
                text = element.string.strip()
                if text:
                    state["buffer"].append(text)
                return
                
            # Otherwise, traverse children
            for child in element.children:
                if child.name:
                    walk(child)
                elif isinstance(child, str):
                    text = child.strip()
                    if text:
                        state["buffer"].append(text)
                        
        walk(body)
        
        # Flush final buffer
        if state["buffer"]:
            text = "\n".join(state["buffer"]).strip()
            if text:
                state["fragments"].append({
                    "text": text,
                    "section_heading": state["heading"],
                    "page_number": 1
                })
                
        return state["fragments"]

class PDFParser(BaseParser):
    """Parses PDF documents, outputting text pages with page numbers."""
    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        fragments = []
        reader = PdfReader(file_path)
        
        for i, page in enumerate(reader.pages):
            page_number = i + 1
            text = page.extract_text()
            if not text or not text.strip():
                continue
                
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            first_line = lines[0] if lines else ""
            # Fallback heading is the first line if it is reasonably short, else 'Page N'
            heading = first_line if (first_line and len(first_line) < 60) else f"Page {page_number}"
            
            fragments.append({
                "text": text.strip(),
                "section_heading": heading,
                "page_number": page_number
            })
            
        return fragments

class DocumentLoader:
    """Manages raw document ingestion, normalization parsers, and storage metadata."""
    def __init__(self, raw_dir: Optional[Path] = None, processed_dir: Optional[Path] = None):
        data_root = settings.get_absolute_path(settings.data_dir)
        self.raw_dir = raw_dir or (data_root / "raw")
        self.processed_dir = processed_dir or (data_root / "processed")
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.parsers: Dict[str, BaseParser] = {
            ".txt": TextParser(),
            ".md": MarkdownParser(),
            ".html": HTMLParser(),
            ".htm": HTMLParser(),
            ".pdf": PDFParser()
        }
        
    def _get_parser(self, extension: str) -> BaseParser:
        parser = self.parsers.get(extension.lower())
        if not parser:
            raise ValueError(f"Unsupported file extension: {extension}")
        return parser
        
    def ingest_file(self, file_path: Path) -> List[DocumentFragment]:
        """
        Ingests a single raw document:
        1. Copies raw document to raw directory.
        2. Parses it to clean plaintext and metadata.
        3. Saves metadata as a processed JSON file for rapid re-indexing.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = path.suffix.lower()
        parser = self._get_parser(ext)
        
        # 1. Copy raw file to raw storage
        raw_dest = self.raw_dir / path.name
        if not raw_dest.exists() or not os.path.samefile(path, raw_dest):
            shutil.copy2(path, raw_dest)
            
        # 2. Parse content
        parsed_sections = parser.parse(raw_dest)
        
        # 3. Normalize into DocumentFragment objects
        fragments = []
        for section in parsed_sections:
            fragments.append(DocumentFragment(
                text=section["text"],
                source_file=path.name,
                section_heading=section.get("section_heading"),
                page_number=section.get("page_number"),
                file_type=ext.lstrip(".")
            ))
            
        # 4. Save processed output to JSON
        processed_data = {
            "source_file": path.name,
            "file_type": ext.lstrip("."),
            "fragments": [frag.model_dump() for frag in fragments]
        }
        
        processed_dest = self.processed_dir / f"{path.stem}.json"
        with open(processed_dest, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
        return fragments

    def load_processed_fragments(self) -> List[DocumentFragment]:
        """Loads all parsed document fragments from the processed JSON files (re-indexing)."""
        fragments = []
        for file in self.processed_dir.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data.get("fragments", []):
                    fragments.append(DocumentFragment(**item))
        return fragments

    def clear_database(self):
        """Cleans out both raw and processed directories."""
        for file in self.raw_dir.glob("*"):
            if file.name != ".gitkeep":
                file.unlink()
        for file in self.processed_dir.glob("*.json"):
            file.unlink()

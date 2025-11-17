"""Multi-strategy PDF parser with table extraction."""

import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

import fitz  # PyMuPDF
import pdfplumber
import pandas as pd

from .document import Document, DocumentChunk, SourceType, ContentType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PDFParser:
    """Parse PDF documents using multiple strategies.

    Tries different parsers in order: PyMuPDF -> pdfplumber -> Tabula
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize PDF parser.

        Args:
            config: PDF processing configuration
        """
        self.config = config or {}
        self.parsers = self.config.get('parsers', ['pymupdf', 'pdfplumber'])
        self.extract_tables = self.config.get('table_extraction', {}).get('enabled', True)

    def parse(self, pdf_path: str) -> Document:
        """Parse PDF file into a Document.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Parsed Document

        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If parsing fails
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("parsing_pdf", path=pdf_path)

        # Try parsers in order
        content = None
        metadata = {}

        for parser in self.parsers:
            try:
                if parser == 'pymupdf':
                    content, metadata = self._parse_with_pymupdf(pdf_file)
                elif parser == 'pdfplumber':
                    content, metadata = self._parse_with_pdfplumber(pdf_file)

                if content:
                    logger.info("parser_success", parser=parser, path=pdf_path)
                    break
            except Exception as e:
                logger.warning("parser_failed", parser=parser, error=str(e))
                continue

        if not content:
            raise ValueError(f"All parsers failed for: {pdf_path}")

        # Create document
        doc_id = str(uuid.uuid4())
        doc = Document(
            doc_id=doc_id,
            content=content,
            source_type=SourceType.PDF,
            title=pdf_file.stem,
            source_path=str(pdf_file.absolute()),
            metadata=metadata
        )

        return doc

    def _parse_with_pymupdf(self, pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse PDF with PyMuPDF (fast, good for text).

        Args:
            pdf_path: Path to PDF

        Returns:
            Tuple of (content, metadata)
        """
        doc = fitz.open(pdf_path)
        pages = []
        metadata = {
            'num_pages': len(doc),
            'pdf_metadata': doc.metadata,
            'parser': 'pymupdf'
        }

        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                pages.append({
                    'page_num': page_num,
                    'text': text,
                    'type': 'text'
                })

        # Combine all pages
        full_text = "\n\n".join(p['text'] for p in pages)

        # Store page info in metadata
        metadata['pages'] = pages

        doc.close()
        return full_text, metadata

    def _parse_with_pdfplumber(self, pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse PDF with pdfplumber (better for tables).

        Args:
            pdf_path: Path to PDF

        Returns:
            Tuple of (content, metadata)
        """
        pages = []
        tables_data = []

        with pdfplumber.open(pdf_path) as pdf:
            metadata = {
                'num_pages': len(pdf.pages),
                'parser': 'pdfplumber'
            }

            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text()

                # Extract tables if enabled
                tables = []
                if self.extract_tables:
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if table:
                            # Convert to DataFrame for better handling
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append({
                                'page': page_num,
                                'table_idx': table_idx,
                                'data': df.to_dict(),
                                'text': df.to_string()
                            })
                            tables_data.append({
                                'page': page_num,
                                'table': df
                            })

                pages.append({
                    'page_num': page_num,
                    'text': text or '',
                    'tables': tables,
                    'type': 'mixed' if tables else 'text'
                })

        # Combine all content
        full_text = "\n\n".join(p['text'] for p in pages)

        metadata['pages'] = pages
        metadata['tables'] = tables_data

        return full_text, metadata

    def extract_chunks(
        self,
        doc: Document,
        chunk_size: int = 512,
        preserve_tables: bool = True
    ) -> List[DocumentChunk]:
        """Extract chunks from document with special table handling.

        Args:
            doc: Parsed document
            chunk_size: Maximum chunk size in characters
            preserve_tables: Don't split tables across chunks

        Returns:
            List of document chunks
        """
        chunks = []
        pages = doc.metadata.get('pages', [])

        for page_info in pages:
            page_num = page_info['page_num']
            text = page_info['text']

            # Handle tables separately
            if preserve_tables and page_info.get('tables'):
                # Create chunks for text
                text_chunks = self._chunk_text(text, chunk_size)
                for idx, chunk_text in enumerate(text_chunks):
                    chunk = self._create_chunk(
                        doc=doc,
                        content=chunk_text,
                        page=page_num,
                        content_type=ContentType.PARAGRAPH,
                        chunk_idx=idx
                    )
                    chunks.append(chunk)

                # Create separate chunks for tables
                for table_info in page_info['tables']:
                    table_text = table_info['text']
                    chunk = self._create_chunk(
                        doc=doc,
                        content=table_text,
                        page=page_num,
                        content_type=ContentType.TABLE,
                        chunk_idx=table_info['table_idx']
                    )
                    chunks.append(chunk)
            else:
                # Regular chunking
                text_chunks = self._chunk_text(text, chunk_size)
                for idx, chunk_text in enumerate(text_chunks):
                    chunk = self._create_chunk(
                        doc=doc,
                        content=chunk_text,
                        page=page_num,
                        content_type=self._detect_content_type(chunk_text),
                        chunk_idx=idx
                    )
                    chunks.append(chunk)

        logger.info("chunks_extracted", num_chunks=len(chunks), doc_id=doc.doc_id)
        return chunks

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks.

        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size

        Returns:
            List of text chunks
        """
        # Use centralized chunking utility to avoid code duplication
        from ..chunking.chunker import chunk_text_simple
        return chunk_text_simple(text, chunk_size=chunk_size)

    def _create_chunk(
        self,
        doc: Document,
        content: str,
        page: int,
        content_type: ContentType,
        chunk_idx: int
    ) -> DocumentChunk:
        """Create a DocumentChunk.

        Args:
            doc: Source document
            content: Chunk content
            page: Page number
            content_type: Type of content
            chunk_idx: Index of chunk on page

        Returns:
            DocumentChunk
        """
        chunk_id = f"{doc.doc_id}_p{page}_c{chunk_idx}"

        return DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            doc_id=doc.doc_id,
            source_type=SourceType.PDF,
            content_type=content_type,
            page=page,
            source_doc=doc.title,
            source_path=doc.source_path,
            char_count=len(content),
            metadata={'chunk_idx': chunk_idx}
        )

    def _detect_content_type(self, text: str) -> ContentType:
        """Detect content type from text.

        Args:
            text: Text to analyze

        Returns:
            Detected ContentType
        """
        # Check for legal references
        if re.search(r'§\s*\d+', text):
            return ContentType.LEGAL_REFERENCE

        # Check for lists
        if re.search(r'^\s*[-•*]\s+', text, re.MULTILINE):
            return ContentType.LIST

        # Check for headings (short text, possibly numbered)
        if len(text) < 100 and re.match(r'^[\d.]+\s+[A-ZÄÖÜ]', text):
            return ContentType.HEADING

        # Default to paragraph
        return ContentType.PARAGRAPH

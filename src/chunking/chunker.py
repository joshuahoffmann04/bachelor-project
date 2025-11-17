"""Hybrid chunking with semantic and sliding window strategies."""

import re
import uuid
from enum import Enum
from typing import List, Dict, Any, Optional

import tiktoken

from ..preprocessing.document import Document, DocumentChunk, ContentType, SourceType
from ..utils.logger import get_logger

logger = get_logger(__name__)


def chunk_text_simple(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Simple text chunking utility function.

    Splits text into chunks by paragraphs, ensuring chunks don't exceed chunk_size.
    This is a lightweight utility for basic chunking needs.

    Args:
        text: Text to chunk
        chunk_size: Maximum chunk size in characters
        overlap: Character overlap between chunks (not used in paragraph mode, reserved for future)

    Returns:
        List of text chunks

    Examples:
        >>> text = "Para 1\\n\\nPara 2\\n\\nPara 3"
        >>> chunks = chunk_text_simple(text, chunk_size=100)
        >>> len(chunks) >= 1
        True

    Note:
        This function is used by pdf_parser.py and web_scraper.py for simple chunking.
        For more advanced chunking with semantic analysis, use HybridChunker class.
    """
    # Split by paragraphs first (double newlines)
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_size = len(para)

        # If adding this paragraph would exceed chunk_size and we have content, start new chunk
        if current_size + para_size > chunk_size and current_chunk:
            # Save current chunk
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            # Add to current chunk
            current_chunk.append(para)
            current_size += para_size

    # Add final chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


class ChunkingStrategy(str, Enum):
    """Chunking strategy types."""
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"
    HYBRID = "hybrid"


class HybridChunker:
    """Hybrid chunking with semantic and sliding window strategies.

    Tries semantic chunking first (by paragraphs/sections),
    falls back to sliding window for large blocks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config or {}

        # Strategy
        strategy = self.config.get('strategy', 'hybrid')
        self.strategy = ChunkingStrategy(strategy)

        # Semantic chunking settings
        semantic_config = self.config.get('semantic', {})
        self.min_chunk_size = semantic_config.get('min_chunk_size', 100)
        self.max_chunk_size = semantic_config.get('max_chunk_size', 512)

        # Sliding window settings
        window_config = self.config.get('sliding_window', {})
        self.window_size = window_config.get('chunk_size', 512)
        self.window_overlap = window_config.get('overlap', 50)

        # Special handling
        special_config = self.config.get('special_handling', {})
        self.preserve_tables = special_config.get('tables', {}).get('split', False) == False
        self.table_max_size = special_config.get('tables', {}).get('max_size', 1024)
        self.keep_lists_together = special_config.get('lists', {}).get('keep_together', True)

        # Tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
            logger.warning("tiktoken_unavailable", msg="Using character count instead")

    def chunk_document(self, doc: Document) -> List[DocumentChunk]:
        """Chunk a document.

        Args:
            doc: Document to chunk

        Returns:
            List of document chunks
        """
        logger.info("chunking_document", doc_id=doc.doc_id, strategy=self.strategy.value)

        if self.strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._semantic_chunking(doc)
        elif self.strategy == ChunkingStrategy.SLIDING_WINDOW:
            chunks = self._sliding_window_chunking(doc)
        else:  # HYBRID
            chunks = self._hybrid_chunking(doc)

        # Count tokens for all chunks
        for chunk in chunks:
            chunk.tokens = self._count_tokens(chunk.content)

        logger.info("chunking_complete", doc_id=doc.doc_id, num_chunks=len(chunks))
        return chunks

    def _hybrid_chunking(self, doc: Document) -> List[DocumentChunk]:
        """Hybrid chunking: semantic first, sliding window fallback.

        Args:
            doc: Document to chunk

        Returns:
            List of chunks
        """
        # Try semantic chunking first
        semantic_chunks = self._semantic_chunking(doc)

        # Check if any chunks are too large
        final_chunks = []
        for chunk in semantic_chunks:
            if len(chunk.content) > self.max_chunk_size:
                # Use sliding window on this chunk
                sub_chunks = self._sliding_window_on_text(
                    chunk.content,
                    doc=doc,
                    parent_chunk=chunk
                )
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _semantic_chunking(self, doc: Document) -> List[DocumentChunk]:
        """Semantic chunking based on document structure.

        Args:
            doc: Document to chunk

        Returns:
            List of chunks
        """
        chunks = []

        # Split by double newlines (paragraphs)
        paragraphs = doc.content.split('\n\n')

        current_chunk_text = []
        current_chunk_size = 0
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Detect content type
            content_type = self._detect_content_type(para)

            # Special handling for tables
            if content_type == ContentType.TABLE and self.preserve_tables:
                # Save current chunk if any
                if current_chunk_text:
                    chunk = self._create_chunk(
                        doc=doc,
                        content='\n\n'.join(current_chunk_text),
                        chunk_idx=chunk_idx,
                        content_type=ContentType.PARAGRAPH
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                    current_chunk_text = []
                    current_chunk_size = 0

                # Create separate chunk for table
                chunk = self._create_chunk(
                    doc=doc,
                    content=para,
                    chunk_idx=chunk_idx,
                    content_type=ContentType.TABLE
                )
                chunks.append(chunk)
                chunk_idx += 1
                continue

            # Add to current chunk or start new one
            para_size = len(para)

            if current_chunk_size + para_size > self.max_chunk_size and current_chunk_text:
                # Save current chunk
                chunk = self._create_chunk(
                    doc=doc,
                    content='\n\n'.join(current_chunk_text),
                    chunk_idx=chunk_idx,
                    content_type=ContentType.PARAGRAPH
                )
                chunks.append(chunk)
                chunk_idx += 1

                # Start new chunk
                current_chunk_text = [para]
                current_chunk_size = para_size
            else:
                current_chunk_text.append(para)
                current_chunk_size += para_size

        # Add final chunk
        if current_chunk_text:
            chunk = self._create_chunk(
                doc=doc,
                content='\n\n'.join(current_chunk_text),
                chunk_idx=chunk_idx,
                content_type=ContentType.PARAGRAPH
            )
            chunks.append(chunk)

        return chunks

    def _sliding_window_chunking(self, doc: Document) -> List[DocumentChunk]:
        """Sliding window chunking.

        Args:
            doc: Document to chunk

        Returns:
            List of chunks
        """
        return self._sliding_window_on_text(doc.content, doc=doc)

    def _sliding_window_on_text(
        self,
        text: str,
        doc: Document,
        parent_chunk: Optional[DocumentChunk] = None
    ) -> List[DocumentChunk]:
        """Apply sliding window to text.

        Args:
            text: Text to chunk
            doc: Source document
            parent_chunk: Optional parent chunk for metadata

        Returns:
            List of chunks
        """
        chunks = []
        words = text.split()

        # Calculate approximate words per chunk
        # Assume average word length of 5 chars
        words_per_chunk = self.window_size // 6
        overlap_words = self.window_overlap // 6

        start = 0
        chunk_idx = 0

        while start < len(words):
            end = start + words_per_chunk
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            # Create chunk
            chunk = self._create_chunk(
                doc=doc,
                content=chunk_text,
                chunk_idx=chunk_idx,
                content_type=ContentType.PARAGRAPH,
                parent_chunk=parent_chunk
            )
            chunks.append(chunk)

            chunk_idx += 1
            start = end - overlap_words

            # Avoid infinite loop
            if end >= len(words):
                break

        return chunks

    def _create_chunk(
        self,
        doc: Document,
        content: str,
        chunk_idx: int,
        content_type: ContentType,
        parent_chunk: Optional[DocumentChunk] = None
    ) -> DocumentChunk:
        """Create a document chunk.

        Args:
            doc: Source document
            content: Chunk content
            chunk_idx: Chunk index
            content_type: Type of content
            parent_chunk: Optional parent chunk

        Returns:
            DocumentChunk
        """
        # Generate chunk ID
        chunk_id = f"{doc.doc_id}_c{chunk_idx}"
        if parent_chunk:
            chunk_id = f"{parent_chunk.chunk_id}_s{chunk_idx}"

        # Extract metadata from parent chunk if available
        page = parent_chunk.page if parent_chunk else None
        section = parent_chunk.section if parent_chunk else None
        source_doc = parent_chunk.source_doc if parent_chunk else doc.title
        source_url = parent_chunk.source_url if parent_chunk else doc.source_url
        source_path = parent_chunk.source_path if parent_chunk else doc.source_path
        scraped_at = parent_chunk.scraped_at if parent_chunk else doc.scraped_at

        return DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            doc_id=doc.doc_id,
            source_type=doc.source_type,
            content_type=content_type,
            page=page,
            section=section,
            source_doc=source_doc,
            source_url=source_url,
            source_path=source_path,
            scraped_at=scraped_at,
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
        # Check for tables (multiple | or tab separators)
        if text.count('|') >= 3 or text.count('\t') >= 3:
            return ContentType.TABLE

        # Check for legal references
        if re.search(r'§\s*\d+', text):
            return ContentType.LEGAL_REFERENCE

        # Check for lists
        if re.search(r'^\s*[-•*]\s+', text, re.MULTILINE) or \
           re.search(r'^\s*\d+\.\s+', text, re.MULTILINE):
            return ContentType.LIST

        # Check for headings
        if len(text) < 100 and re.match(r'^[\d.]+\s+[A-ZÄÖÜ]', text):
            return ContentType.HEADING

        # Default
        return ContentType.PARAGRAPH

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass

        # Fallback: approximate as words * 1.3
        return int(len(text.split()) * 1.3)

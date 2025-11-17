"""Document data structures."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    """Document source type."""
    PDF = "pdf"
    WEB = "web"
    HTML = "html"
    TEXT = "text"


class ContentType(str, Enum):
    """Content type within document."""
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    HEADING = "heading"
    LEGAL_REFERENCE = "legal_reference"
    CODE = "code"
    UNKNOWN = "unknown"


@dataclass
class Document:
    """Represents a source document."""

    doc_id: str
    content: str
    source_type: SourceType
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional fields
    title: Optional[str] = None
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    scraped_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate and normalize fields."""
        if isinstance(self.source_type, str):
            self.source_type = SourceType(self.source_type)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document for RAG."""

    chunk_id: str
    content: str
    doc_id: str
    source_type: SourceType
    content_type: ContentType

    # Location information
    page: Optional[int] = None
    section: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    # Source information
    source_doc: Optional[str] = None
    source_url: Optional[str] = None
    source_path: Optional[str] = None

    # Processing metadata
    tokens: int = 0
    char_count: int = 0
    scraped_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Embedding (populated during indexing)
    embedding: Optional[List[float]] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived fields."""
        if isinstance(self.source_type, str):
            self.source_type = SourceType(self.source_type)
        if isinstance(self.content_type, str):
            self.content_type = ContentType(self.content_type)

        if not self.char_count:
            self.char_count = len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'doc_id': self.doc_id,
            'source_type': self.source_type.value,
            'content_type': self.content_type.value,
            'page': self.page,
            'section': self.section,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'source_doc': self.source_doc,
            'source_url': self.source_url,
            'source_path': self.source_path,
            'tokens': self.tokens,
            'char_count': self.char_count,
            'scraped_at': self.scraped_at.isoformat() if self.scraped_at else None,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create from dictionary."""
        # Parse datetime fields
        if data.get('scraped_at'):
            data['scraped_at'] = datetime.fromisoformat(data['scraped_at'])
        if data.get('created_at'):
            data['created_at'] = datetime.fromisoformat(data['created_at'])

        return cls(**data)

    def get_citation(self) -> str:
        """Generate citation string for this chunk.

        Returns:
            Formatted citation string
        """
        parts = []

        # Add source document
        if self.source_doc:
            parts.append(f"Quelle: {self.source_doc}")
        elif self.source_url:
            parts.append(f"Quelle: {self.source_url}")

        # Add page or section
        if self.page is not None:
            parts.append(f"Seite {self.page}")
        if self.section:
            parts.append(self.section)

        # Add scraping timestamp for web sources
        if self.source_type == SourceType.WEB and self.scraped_at:
            date_str = self.scraped_at.strftime("%Y-%m-%d")
            parts.append(f"abgerufen am {date_str}")

        return ", ".join(parts) if parts else "Quelle: Unbekannt"

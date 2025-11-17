"""Unit tests for chunking module."""

import pytest
from src.chunking.chunker import chunk_text_simple, HybridChunker, ChunkingStrategy
from src.preprocessing.document import Document, SourceType


@pytest.mark.unit
class TestChunkTextSimple:
    """Tests for chunk_text_simple utility function."""

    def test_basic_chunking(self):
        """Test basic text chunking."""
        text = "Para 1\n\nPara 2\n\nPara 3"
        chunks = chunk_text_simple(text, chunk_size=50)

        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_respects_chunk_size(self):
        """Test that chunks respect maximum size."""
        text = "\n\n".join([f"Paragraph {i}" * 20 for i in range(10)])
        chunk_size = 100
        chunks = chunk_text_simple(text, chunk_size=chunk_size)

        # Most chunks should be under the limit
        # (some may slightly exceed due to paragraph boundaries)
        oversized = [c for c in chunks if len(c) > chunk_size * 1.5]
        assert len(oversized) < len(chunks) * 0.3  # Less than 30% oversized

    def test_empty_text(self):
        """Test chunking empty text."""
        chunks = chunk_text_simple("", chunk_size=100)
        assert chunks == []

    def test_single_paragraph(self):
        """Test chunking single paragraph."""
        text = "Dies ist ein einzelner Absatz ohne Zeilenumbrüche."
        chunks = chunk_text_simple(text, chunk_size=100)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_preserves_paragraph_boundaries(self):
        """Test that paragraph boundaries are preserved."""
        text = "Para 1 short\n\nPara 2 also short\n\nPara 3 short"
        chunks = chunk_text_simple(text, chunk_size=100)

        # All paragraphs should fit in one chunk
        assert len(chunks) == 1
        assert "Para 1" in chunks[0]
        assert "Para 2" in chunks[0]
        assert "Para 3" in chunks[0]


@pytest.mark.unit
class TestHybridChunker:
    """Tests for HybridChunker class."""

    def test_initialization(self, sample_config):
        """Test chunker initialization."""
        chunker = HybridChunker(sample_config['chunking'])

        assert chunker.strategy == ChunkingStrategy.HYBRID
        assert chunker.max_chunk_size == 512
        assert chunker.window_size == 512

    def test_semantic_chunking(self, sample_config, sample_document):
        """Test semantic chunking strategy."""
        config = sample_config['chunking'].copy()
        config['strategy'] = 'semantic'

        chunker = HybridChunker(config)
        chunks = chunker.chunk_document(sample_document)

        assert len(chunks) > 0
        assert all(chunk.doc_id == sample_document.doc_id for chunk in chunks)
        assert all(chunk.tokens > 0 for chunk in chunks)

    def test_sliding_window_chunking(self, sample_config, sample_document):
        """Test sliding window chunking strategy."""
        config = sample_config['chunking'].copy()
        config['strategy'] = 'sliding_window'

        chunker = HybridChunker(config)
        chunks = chunker.chunk_document(sample_document)

        assert len(chunks) > 0
        assert all(chunk.doc_id == sample_document.doc_id for chunk in chunks)

    def test_hybrid_chunking(self, sample_config, sample_document):
        """Test hybrid chunking strategy."""
        chunker = HybridChunker(sample_config['chunking'])
        chunks = chunker.chunk_document(sample_document)

        assert len(chunks) > 0
        assert all(chunk.doc_id == sample_document.doc_id for chunk in chunks)
        # Verify chunk IDs are sequential
        assert chunks[0].chunk_id.endswith('_c0')

    def test_chunk_metadata(self, sample_config, sample_document):
        """Test that chunks preserve document metadata."""
        chunker = HybridChunker(sample_config['chunking'])
        chunks = chunker.chunk_document(sample_document)

        for chunk in chunks:
            assert chunk.source_type == sample_document.source_type
            assert chunk.char_count == len(chunk.content)
            assert chunk.tokens > 0

    def test_long_document_chunking(self, sample_config, sample_text_long):
        """Test chunking of long documents."""
        from src.preprocessing.document import Document, SourceType

        doc = Document(
            doc_id="long_doc",
            title="Long Document",
            content=sample_text_long,
            source_type=SourceType.PDF
        )

        chunker = HybridChunker(sample_config['chunking'])
        chunks = chunker.chunk_document(doc)

        # Long document should create multiple chunks
        assert len(chunks) > 3
        # All chunks should have reasonable size
        assert all(100 < len(c.content) < 1000 for c in chunks)


@pytest.mark.unit
def test_token_counting(sample_config):
    """Test token counting functionality."""
    chunker = HybridChunker(sample_config['chunking'])

    text = "Dies ist ein Test mit mehreren Wörtern."
    token_count = chunker._count_tokens(text)

    # Should return positive count
    assert token_count > 0
    # Should be roughly proportional to word count
    word_count = len(text.split())
    assert token_count >= word_count  # At least as many tokens as words

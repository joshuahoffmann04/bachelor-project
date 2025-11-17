"""Sparse retrieval using BM25 keyword search."""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from rank_bm25 import BM25Okapi

from ..preprocessing.document import DocumentChunk
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SparseRetriever:
    """Sparse retrieval using BM25 algorithm."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize sparse retriever.

        Args:
            config: Retrieval configuration
        """
        self.config = config or {}

        # BM25 parameters
        retrieval_config = self.config.get('retrieval', {}).get('sparse', {})
        self.k1 = retrieval_config.get('k1', 1.5)
        self.b = retrieval_config.get('b', 0.75)
        self.top_k = retrieval_config.get('top_k', 10)

        # Index
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Build BM25 index from document chunks.

        Args:
            chunks: List of document chunks to index
        """
        logger.info("building_bm25_index", num_chunks=len(chunks))

        self.chunks = chunks

        # Tokenize corpus
        self.tokenized_corpus = [
            self._tokenize(chunk.content)
            for chunk in chunks
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )

        logger.info("bm25_index_built", num_docs=len(self.tokenized_corpus))

    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[DocumentChunk, float]]:
        """Search for relevant chunks using BM25.

        Args:
            query: Search query
            top_k: Number of results to return (overrides default)

        Returns:
            List of (chunk, score) tuples, sorted by relevance
        """
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        k = top_k or self.top_k

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        # Create results
        results = [
            (self.chunks[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0  # Only return non-zero scores
        ]

        logger.info("bm25_search_complete", query_len=len(query), num_results=len(results))
        return results

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25.

        Simple word tokenization with lowercase normalization.
        Preserves special characters like § for legal references.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()

        # Split on whitespace
        tokens = text.split()

        # Basic cleaning while preserving important characters
        cleaned_tokens = []
        for token in tokens:
            # Remove pure punctuation tokens, but keep § and digits
            if token.strip() and not token.strip('.,;:!?'):
                continue
            cleaned_tokens.append(token)

        return cleaned_tokens

    def save_index(self, index_path: str) -> None:
        """Save BM25 index to disk.

        Args:
            index_path: Path to save index
        """
        if self.bm25 is None:
            raise RuntimeError("No index to save")

        index_file = Path(index_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)

        # Save all necessary data
        data = {
            'bm25': self.bm25,
            'chunks': self.chunks,
            'tokenized_corpus': self.tokenized_corpus,
            'k1': self.k1,
            'b': self.b
        }

        with open(index_file, 'wb') as f:
            pickle.dump(data, f)

        logger.info("bm25_index_saved", path=index_path)

    def load_index(self, index_path: str) -> None:
        """Load BM25 index from disk.

        Args:
            index_path: Path to index file
        """
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index not found: {index_path}")

        with open(index_path, 'rb') as f:
            data = pickle.load(f)

        self.bm25 = data['bm25']
        self.chunks = data['chunks']
        self.tokenized_corpus = data['tokenized_corpus']
        self.k1 = data['k1']
        self.b = data['b']

        logger.info("bm25_index_loaded", path=index_path, num_chunks=len(self.chunks))

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics.

        Returns:
            Dictionary with stats
        """
        avg_doc_len = sum(len(doc) for doc in self.tokenized_corpus) / len(self.tokenized_corpus) \
            if self.tokenized_corpus else 0

        return {
            'num_chunks': len(self.chunks),
            'num_docs': len(self.tokenized_corpus),
            'avg_doc_length': avg_doc_len,
            'k1': self.k1,
            'b': self.b,
            'top_k': self.top_k
        }

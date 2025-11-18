"""Hybrid retrieval combining dense and sparse methods."""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever
from ..preprocessing.document import DocumentChunk
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Hybrid retriever combining dense (FAISS) and sparse (BM25) retrieval.

    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize hybrid retriever.

        Args:
            config: Retrieval configuration
        """
        self.config = config or {}

        # Create retrievers
        self.dense_retriever = DenseRetriever(config)
        self.sparse_retriever = SparseRetriever(config)

        # Fusion settings
        fusion_config = self.config.get('retrieval', {}).get('hybrid', {})
        self.dense_weight = fusion_config.get('dense_weight', 0.7)
        self.sparse_weight = fusion_config.get('sparse_weight', 0.3)
        self.fusion_method = fusion_config.get('fusion_method', 'rrf')

        # Final selection
        final_config = self.config.get('retrieval', {})
        self.final_top_k = final_config.get('final_top_k', 5)

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Build both dense and sparse indices.

        Args:
            chunks: List of document chunks to index
        """
        logger.info("building_hybrid_index", num_chunks=len(chunks))

        # Build both indices
        self.dense_retriever.build_index(chunks)
        self.sparse_retriever.build_index(chunks)

        logger.info("hybrid_index_built")

    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[DocumentChunk, float]]:
        """Search using hybrid retrieval.

        Args:
            query: Search query
            top_k: Number of results to return (overrides default)

        Returns:
            List of (chunk, score) tuples, sorted by relevance
        """
        k = top_k or self.final_top_k

        # Get results from both retrievers
        dense_results = self.dense_retriever.search(query, top_k=k * 2)
        sparse_results = self.sparse_retriever.search(query, top_k=k * 2)

        # Fuse results
        if self.fusion_method == 'rrf':
            fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)
        else:
            fused_results = self._weighted_fusion(dense_results, sparse_results)

        # Return top-k
        final_results = fused_results[:k]

        logger.info(
            "hybrid_search_complete",
            query_len=len(query),
            dense_results=len(dense_results),
            sparse_results=len(sparse_results),
            final_results=len(final_results)
        )

        return final_results

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
        k: int = 60
    ) -> List[Tuple[DocumentChunk, float]]:
        """Combine results using Reciprocal Rank Fusion.

        RRF formula: score = sum(1 / (k + rank))

        Args:
            dense_results: Results from dense retriever
            sparse_results: Results from sparse retriever
            k: RRF constant (default 60)

        Returns:
            Fused and sorted results
        """
        # Calculate RRF scores
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, DocumentChunk] = {}

        # Process dense results
        for rank, (chunk, score) in enumerate(dense_results, 1):
            rrf_score = self.dense_weight / (k + rank)
            rrf_scores[chunk.chunk_id] += rrf_score
            chunk_map[chunk.chunk_id] = chunk

        # Process sparse results
        for rank, (chunk, score) in enumerate(sparse_results, 1):
            rrf_score = self.sparse_weight / (k + rank)
            rrf_scores[chunk.chunk_id] += rrf_score
            chunk_map[chunk.chunk_id] = chunk

        # Sort by RRF score
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Normalize scores to [0, 1] range for better interpretability
        if sorted_chunks:
            scores = [score for _, score in sorted_chunks]
            min_score = min(scores)
            max_score = max(scores)

            if max_score > min_score:
                # Min-max normalization
                normalized_chunks = [
                    (chunk_id, (score - min_score) / (max_score - min_score))
                    for chunk_id, score in sorted_chunks
                ]
            else:
                # All scores are the same
                normalized_chunks = [(chunk_id, 1.0) for chunk_id, _ in sorted_chunks]
        else:
            normalized_chunks = []

        # Create results with normalized scores
        results = [
            (chunk_map[chunk_id], score)
            for chunk_id, score in normalized_chunks
        ]

        return results

    def _weighted_fusion(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]]
    ) -> List[Tuple[DocumentChunk, float]]:
        """Combine results using weighted score fusion.

        Args:
            dense_results: Results from dense retriever
            sparse_results: Results from sparse retriever

        Returns:
            Fused and sorted results
        """
        # Normalize scores
        dense_normalized = self._normalize_scores(dense_results)
        sparse_normalized = self._normalize_scores(sparse_results)

        # Combine scores
        combined_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, DocumentChunk] = {}

        # Add dense scores
        for chunk, score in dense_normalized:
            combined_scores[chunk.chunk_id] += self.dense_weight * score
            chunk_map[chunk.chunk_id] = chunk

        # Add sparse scores
        for chunk, score in sparse_normalized:
            combined_scores[chunk.chunk_id] += self.sparse_weight * score
            chunk_map[chunk.chunk_id] = chunk

        # Sort by combined score
        sorted_chunks = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Create results
        results = [
            (chunk_map[chunk_id], score)
            for chunk_id, score in sorted_chunks
        ]

        return results

    def _normalize_scores(
        self,
        results: List[Tuple[DocumentChunk, float]]
    ) -> List[Tuple[DocumentChunk, float]]:
        """Normalize scores to [0, 1] range.

        Args:
            results: List of (chunk, score) tuples

        Returns:
            Normalized results
        """
        if not results:
            return []

        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores are the same
            return [(chunk, 1.0) for chunk, _ in results]

        # Min-max normalization
        normalized = [
            (chunk, (score - min_score) / (max_score - min_score))
            for chunk, score in results
        ]

        return normalized

    def save_indices(self, dense_index_path: str, dense_chunks_path: str, sparse_index_path: str) -> None:
        """Save both indices.

        Args:
            dense_index_path: Path for FAISS index
            dense_chunks_path: Path for chunks
            sparse_index_path: Path for BM25 index
        """
        self.dense_retriever.save_index(dense_index_path, dense_chunks_path)
        self.sparse_retriever.save_index(sparse_index_path)
        logger.info("hybrid_indices_saved")

    def load_indices(self, dense_index_path: str, dense_chunks_path: str, sparse_index_path: str) -> None:
        """Load both indices.

        Args:
            dense_index_path: Path to FAISS index
            dense_chunks_path: Path to chunks
            sparse_index_path: Path to BM25 index
        """
        self.dense_retriever.load_index(dense_index_path, dense_chunks_path)
        self.sparse_retriever.load_index(sparse_index_path)
        logger.info("hybrid_indices_loaded")

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics.

        Returns:
            Dictionary with stats
        """
        return {
            'dense': self.dense_retriever.get_stats(),
            'sparse': self.sparse_retriever.get_stats(),
            'fusion': {
                'method': self.fusion_method,
                'dense_weight': self.dense_weight,
                'sparse_weight': self.sparse_weight,
                'final_top_k': self.final_top_k
            }
        }

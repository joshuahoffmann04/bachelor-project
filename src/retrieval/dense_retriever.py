"""Dense retrieval using FAISS and sentence transformers."""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ..preprocessing.document import DocumentChunk
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """Dense retrieval using FAISS vector similarity search."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dense retriever.

        Args:
            config: Retrieval configuration
        """
        self.config = config or {}

        # Embedding model
        embedding_config = self.config.get('embeddings', {})
        model_name = embedding_config.get(
            'model',
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        )
        device = embedding_config.get('device', 'cpu')

        logger.info("loading_embedding_model", model=model_name, device=device)
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Retrieval settings
        retrieval_config = self.config.get('retrieval', {}).get('dense', {})
        self.top_k = retrieval_config.get('top_k', 10)
        self.similarity_threshold = retrieval_config.get('similarity_threshold', 0.5)

        # FAISS index
        self.index: Optional[faiss.Index] = None
        self.chunks: List[DocumentChunk] = []

        # Batch processing
        self.batch_size = embedding_config.get('batch_size', 32)

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Build FAISS index from document chunks.

        Args:
            chunks: List of document chunks to index
        """
        logger.info("building_index", num_chunks=len(chunks))

        self.chunks = chunks

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self._embed_texts(texts)

        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))

        # Store embeddings in chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()

        logger.info("index_built", num_vectors=self.index.ntotal)

    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[DocumentChunk, float]]:
        """Search for relevant chunks.

        Args:
            query: Search query
            top_k: Number of results to return (overrides default)

        Returns:
            List of (chunk, score) tuples, sorted by relevance
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        k = top_k or self.top_k

        # Embed query
        query_embedding = self._embed_texts([query])[0]

        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )

        # Convert distances to similarity scores
        # L2 distance -> similarity (inverse)
        # Lower distance = higher similarity
        max_distance = distances[0].max() if len(distances[0]) > 0 else 1.0
        similarities = 1.0 - (distances[0] / (max_distance + 1e-6))

        # Filter by threshold and create results
        results = []
        for idx, score in zip(indices[0], similarities):
            if score >= self.similarity_threshold:
                chunk = self.chunks[idx]
                results.append((chunk, float(score)))

        logger.info("dense_search_complete", query_len=len(query), num_results=len(results))
        return results

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts using the model.

        Args:
            texts: List of texts to embed

        Returns:
            Numpy array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    def save_index(self, index_path: str, chunks_path: str) -> None:
        """Save FAISS index and chunks to disk.

        Args:
            index_path: Path to save FAISS index
            chunks_path: Path to save chunks
        """
        if self.index is None:
            raise RuntimeError("No index to save")

        # Save FAISS index
        index_file = Path(index_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_file))

        # Save chunks
        chunks_file = Path(chunks_path)
        chunks_file.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_file, 'wb') as f:
            pickle.dump(self.chunks, f)

        logger.info("index_saved", index_path=index_path, chunks_path=chunks_path)

    def load_index(self, index_path: str, chunks_path: str) -> None:
        """Load FAISS index and chunks from disk.

        Args:
            index_path: Path to FAISS index
            chunks_path: Path to chunks
        """
        # Load FAISS index
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index not found: {index_path}")

        self.index = faiss.read_index(index_path)

        # Load chunks
        if not Path(chunks_path).exists():
            raise FileNotFoundError(f"Chunks not found: {chunks_path}")

        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)

        logger.info("index_loaded", index_path=index_path, num_chunks=len(self.chunks))

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics.

        Returns:
            Dictionary with stats
        """
        return {
            'num_chunks': len(self.chunks),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embedding_dim,
            'model_name': self.model._model_name,
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold
        }

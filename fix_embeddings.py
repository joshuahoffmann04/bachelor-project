#!/usr/bin/env python3
"""Quick fix script for embedding issues."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import load_config
from src.retrieval.dense_retriever import DenseRetriever
import pickle

def check_embeddings():
    """Check if chunks have embeddings."""
    print("Checking embeddings...")

    with open('data/processed/chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)

    print(f"Total chunks: {len(chunks)}")

    chunks_with_embeddings = sum(1 for c in chunks if c.embedding is not None)
    chunks_without = len(chunks) - chunks_with_embeddings

    print(f"Chunks with embeddings: {chunks_with_embeddings}")
    print(f"Chunks WITHOUT embeddings: {chunks_without}")

    if chunks_without > 0:
        print("\n❌ PROBLEM: Chunks haben keine Embeddings!")
        print("Fix: python pipeline.py build-index")
        return False

    # Check embedding dimension
    if chunks_with_embeddings > 0:
        sample_embedding = next(c.embedding for c in chunks if c.embedding)
        print(f"Embedding dimension: {len(sample_embedding)}")

    print("\n✓ Embeddings vorhanden")
    return True

def fix_similarity_threshold():
    """Lower similarity threshold."""
    import yaml

    print("\nFixing similarity threshold...")

    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    old_threshold = config.get('retrieval', {}).get('dense', {}).get('similarity_threshold', 0.5)

    if 'retrieval' not in config:
        config['retrieval'] = {}
    if 'dense' not in config['retrieval']:
        config['retrieval']['dense'] = {}

    config['retrieval']['dense']['similarity_threshold'] = 0.0

    with open('config/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Similarity threshold: {old_threshold} → 0.0")
    print("\nJetzt testen: python debug.py test-retrieval 'Algorithmen'")

if __name__ == '__main__':
    has_embeddings = check_embeddings()

    if has_embeddings:
        fix_similarity_threshold()
    else:
        print("\nERSTEL NEUE EMBEDDINGS:")
        print("python pipeline.py build-index")

#!/usr/bin/env python3
"""Debug script to analyze abstaining issue."""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.rag_generator import RAGGenerator


async def debug_retrieval():
    """Debug retrieval scores."""

    # Load config
    config = load_config("config/config.yaml")
    setup_logger(log_level="DEBUG", log_format="console", enable_privacy=False)

    # Initialize retriever
    retriever = HybridRetriever(config)

    # Check if indices exist
    if not Path("data/processed/faiss_index.bin").exists():
        print("❌ FEHLER: Indices nicht gefunden!")
        print("   Bitte zuerst ausführen: python pipeline.py build-index")
        return

    # Load indices
    print("Loading indices...")
    retriever.load_indices(
        dense_index_path="data/processed/faiss_index.bin",
        dense_chunks_path="data/processed/chunks.pkl",
        sparse_index_path="data/processed/bm25_index.pkl"
    )

    # Test query
    test_queries = [
        "Wie viele ECTS hat das Modul Algorithmen?",
        "Welche Voraussetzungen gibt es für die Bachelorarbeit?",
        "Bis wann muss ich mich anmelden?"
    ]

    print("\n" + "="*80)
    print("RETRIEVAL SCORE ANALYSE")
    print("="*80 + "\n")

    for query in test_queries:
        print(f"\n📋 Query: '{query}'")
        print("-" * 80)

        # Search
        results = retriever.search(query, top_k=5)

        if not results:
            print("⚠️  KEINE RESULTS!")
            continue

        # Show scores
        scores = [score for _, score in results]
        avg_score = sum(scores) / len(scores)

        print(f"Anzahl Results: {len(results)}")
        print(f"Scores: {[f'{s:.4f}' for s in scores]}")
        print(f"Average Score: {avg_score:.4f}")
        print(f"Min Score: {min(scores):.4f}")
        print(f"Max Score: {max(scores):.4f}")

        # Check against threshold
        threshold = config.get('prompts', {}).get('abstaining', {}).get('threshold', 0.5)
        print(f"\n🎯 Abstaining Threshold: {threshold}")

        if avg_score < threshold:
            print(f"❌ ABSTAINING würde greifen! ({avg_score:.4f} < {threshold})")
        else:
            print(f"✅ Antwort würde generiert werden ({avg_score:.4f} >= {threshold})")

        # Show top chunks
        print("\nTop 3 Chunks:")
        for idx, (chunk, score) in enumerate(results[:3], 1):
            print(f"\n  [{idx}] Score: {score:.4f}")
            print(f"      Source: {chunk.source_doc or 'Unknown'}")
            print(f"      Type: {chunk.content_type.value}")
            print(f"      Content: {chunk.content[:150]}...")

    # Analyze score distribution
    print("\n" + "="*80)
    print("SCORE STATISTIK")
    print("="*80 + "\n")

    # Get stats
    stats = retriever.get_stats()
    print(f"Dense Retriever:")
    print(f"  - Chunks: {stats['dense'].get('num_chunks', 'N/A')}")
    print(f"  - Model: {stats['dense'].get('model_name', 'N/A')}")

    print(f"\nSparse Retriever:")
    print(f"  - Chunks: {stats['sparse'].get('num_chunks', 'N/A')}")
    print(f"  - k1: {stats['sparse'].get('k1', 'N/A')}")
    print(f"  - b: {stats['sparse'].get('b', 'N/A')}")

    print(f"\nFusion:")
    print(f"  - Method: {stats['fusion']['method']}")
    print(f"  - Dense Weight: {stats['fusion']['dense_weight']}")
    print(f"  - Sparse Weight: {stats['fusion']['sparse_weight']}")


if __name__ == "__main__":
    asyncio.run(debug_retrieval())

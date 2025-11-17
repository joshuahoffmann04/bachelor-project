#!/usr/bin/env python3
"""Example usage of the RAG Studienordnung system."""

import asyncio
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.rag_generator import RAGGenerator


async def main():
    """Example RAG system usage."""

    # 1. Load configuration
    print("Loading configuration...")
    config = load_config("config/config.yaml")

    # 2. Setup logging
    setup_logger(log_level="INFO", log_format="console", enable_privacy=True)

    # 3. Initialize retriever
    print("Initializing retriever...")
    retriever = HybridRetriever(config)

    # Load pre-built indices
    print("Loading indices...")
    retriever.load_indices(
        dense_index_path="data/processed/faiss_index.bin",
        dense_chunks_path="data/processed/chunks.pkl",
        sparse_index_path="data/processed/bm25_index.pkl"
    )

    # 4. Initialize RAG generator
    print("Initializing LLM...")
    generator = RAGGenerator(retriever=retriever, config=config)

    # 5. Ask questions
    questions = [
        "Wie viele ECTS hat das Modul Algorithmen und Datenstrukturen?",
        "Welche Voraussetzungen gibt es für die Bachelorarbeit?",
        "Bis wann muss ich mich für Prüfungen anmelden?",
    ]

    print("\n" + "="*80)
    print("Asking questions...")
    print("="*80 + "\n")

    for question in questions:
        print(f"\nFrage: {question}")
        print("-" * 80)

        # Generate answer
        result = await generator.generate(question)

        # Display results
        print(f"\nAntwort:\n{result['answer']}\n")

        if result['sources']:
            print("Quellen:")
            for idx, source in enumerate(result['sources'], 1):
                print(f"  [{idx}] {source}")

        print(f"\nKonfidenz: {result['confidence']:.2f}")
        print(f"Retrieval-Zeit: {result['metadata']['retrieval_time_ms']:.0f}ms")
        print(f"Generation-Zeit: {result['metadata']['generation_time_ms']:.0f}ms")
        print("=" * 80)


async def retrieval_example():
    """Example of using retrieval only (without LLM)."""

    print("\n" + "="*80)
    print("Retrieval-Only Example")
    print("="*80 + "\n")

    # Load config
    config = load_config("config/config.yaml")

    # Initialize retriever
    retriever = HybridRetriever(config)
    retriever.load_indices(
        dense_index_path="data/processed/faiss_index.bin",
        dense_chunks_path="data/processed/chunks.pkl",
        sparse_index_path="data/processed/bm25_index.pkl"
    )

    # Search
    query = "ECTS Algorithmen"
    results = retriever.search(query, top_k=5)

    print(f"Query: {query}\n")
    print(f"Found {len(results)} results:\n")

    for idx, (chunk, score) in enumerate(results, 1):
        print(f"[{idx}] Score: {score:.3f}")
        print(f"Content: {chunk.content[:200]}...")
        print(f"Source: {chunk.get_citation()}\n")


async def processing_example():
    """Example of processing a new document."""

    print("\n" + "="*80)
    print("Document Processing Example")
    print("="*80 + "\n")

    from src.preprocessing.pdf_parser import PDFParser
    from src.chunking.chunker import HybridChunker

    config = load_config("config/config.yaml")

    # Parse PDF
    parser = PDFParser(config.get('pdf_processing'))
    pdf_path = "data/raw/Pruefungsordnung_BSc_Inf_2024.pdf"

    if Path(pdf_path).exists():
        print(f"Parsing PDF: {pdf_path}")
        doc = parser.parse(pdf_path)

        print(f"✓ Document ID: {doc.doc_id}")
        print(f"✓ Content length: {len(doc.content)} chars")
        print(f"✓ Pages: {doc.metadata.get('num_pages', 'N/A')}")

        # Chunk document
        chunker = HybridChunker(config.get('chunking'))
        chunks = chunker.chunk_document(doc)

        print(f"\n✓ Created {len(chunks)} chunks")
        print(f"\nFirst chunk preview:")
        print(f"  ID: {chunks[0].chunk_id}")
        print(f"  Type: {chunks[0].content_type}")
        print(f"  Tokens: {chunks[0].tokens}")
        print(f"  Content: {chunks[0].content[:200]}...")
    else:
        print(f"PDF not found: {pdf_path}")
        print("Place a PDF in data/raw/ first.")


async def web_scraping_example():
    """Example of web scraping."""

    print("\n" + "="*80)
    print("Web Scraping Example")
    print("="*80 + "\n")

    from src.scraping.web_scraper import WebScraper

    config = load_config("config/config.yaml")
    scraper = WebScraper(config.get('scraping'))

    # Example URL (replace with actual URL from config)
    url = "https://www.mathematik.uni-marburg.de/"

    print(f"Scraping: {url}")
    doc = scraper.scrape_url(url, use_cache=True)

    if doc:
        print(f"✓ Document ID: {doc.doc_id}")
        print(f"✓ Title: {doc.title}")
        print(f"✓ Content length: {len(doc.content)} chars")
        print(f"✓ Scraped at: {doc.scraped_at}")

        # Show metadata
        metadata = doc.metadata
        print(f"\nMetadata:")
        print(f"  Headings: {metadata.get('num_headings', 0)}")
        print(f"  Tables: {metadata.get('num_tables', 0)}")
    else:
        print("Scraping failed")


if __name__ == "__main__":
    print("RAG Studienordnung - Example Usage\n")
    print("Choose an example:")
    print("  1. Full RAG pipeline (retrieval + generation)")
    print("  2. Retrieval only")
    print("  3. Document processing")
    print("  4. Web scraping")
    print("  5. Run all examples")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        asyncio.run(main())
    elif choice == "2":
        asyncio.run(retrieval_example())
    elif choice == "3":
        asyncio.run(processing_example())
    elif choice == "4":
        asyncio.run(web_scraping_example())
    elif choice == "5":
        asyncio.run(retrieval_example())
        asyncio.run(processing_example())
        asyncio.run(web_scraping_example())
        asyncio.run(main())
    else:
        print("Invalid choice")

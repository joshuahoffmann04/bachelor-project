#!/usr/bin/env python3
"""Main pipeline for document processing and indexing."""

import sys
from pathlib import Path
from typing import List

import click
from rich.console import Console
from rich.progress import track

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger, get_logger
from src.preprocessing.pdf_parser import PDFParser
from src.preprocessing.document import Document, DocumentChunk
from src.scraping.web_scraper import WebScraper
from src.chunking.chunker import HybridChunker
from src.retrieval.hybrid_retriever import HybridRetriever

console = Console()
logger = get_logger(__name__)


class ProcessingPipeline:
    """Document processing and indexing pipeline."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize pipeline.

        Args:
            config_path: Path to configuration file
        """
        console.print("[yellow]Loading configuration...[/yellow]")
        self.config = load_config(config_path)

        # Setup logging
        log_config = self.config.get('logging', {})
        setup_logger(
            log_level=log_config.get('level', 'INFO'),
            log_format='console',
            enable_privacy=False  # No privacy needed for pipeline
        )

        # Initialize components
        self.pdf_parser = PDFParser(self.config.get('pdf_processing'))
        self.web_scraper = WebScraper(self.config.get('scraping'))
        self.chunker = HybridChunker(self.config.get('chunking'))
        self.retriever = HybridRetriever(self.config)

        console.print("[green]✓ Pipeline initialized[/green]\n")

    def process_sources(self) -> List[Document]:
        """Process all configured sources.

        Returns:
            List of processed documents
        """
        documents = []
        sources_config = self.config.get('sources', {})

        # Process PDF sources
        for source_name, source_config in sources_config.items():
            if not source_config.get('enabled', True):
                continue

            source_type = source_config.get('type')

            try:
                if source_type == 'pdf':
                    doc = self._process_pdf(source_name, source_config)
                    if doc:
                        documents.append(doc)

                elif source_type == 'web':
                    doc = self._process_web(source_name, source_config)
                    if doc:
                        documents.append(doc)

                else:
                    console.print(f"[yellow]Unknown source type: {source_type}[/yellow]")

            except Exception as e:
                console.print(f"[red]✗ Error processing {source_name}: {e}[/red]")
                logger.error("source_processing_failed", source=source_name, error=str(e))

        console.print(f"\n[green]✓ Processed {len(documents)} documents[/green]")
        return documents

    def _process_pdf(self, source_name: str, source_config: dict) -> Document:
        """Process PDF source.

        Args:
            source_name: Name of source
            source_config: Source configuration

        Returns:
            Processed document
        """
        pdf_path = source_config.get('path')
        console.print(f"[cyan]Processing PDF:[/cyan] {pdf_path}")

        if not Path(pdf_path).exists():
            console.print(f"[yellow]⚠ PDF not found, skipping: {pdf_path}[/yellow]")
            return None

        doc = self.pdf_parser.parse(pdf_path)
        console.print(f"[green]✓ Parsed PDF:[/green] {len(doc.content)} chars")

        return doc

    def _process_web(self, source_name: str, source_config: dict) -> Document:
        """Process web source.

        Args:
            source_name: Name of source
            source_config: Source configuration

        Returns:
            Processed document
        """
        url = source_config.get('url')
        console.print(f"[cyan]Scraping web:[/cyan] {url}")

        use_cache = source_config.get('cache_enabled', True)
        doc = self.web_scraper.scrape_url(url, use_cache=use_cache)

        if doc:
            console.print(f"[green]✓ Scraped:[/green] {len(doc.content)} chars")
        else:
            console.print(f"[yellow]⚠ Scraping failed[/yellow]")

        return doc

    def chunk_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """Chunk all documents.

        Args:
            documents: List of documents

        Returns:
            List of chunks
        """
        console.print("\n[cyan]Chunking documents...[/cyan]")

        all_chunks = []

        for doc in track(documents, description="Chunking..."):
            try:
                chunks = self.chunker.chunk_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                console.print(f"[red]✗ Error chunking {doc.doc_id}: {e}[/red]")
                logger.error("chunking_failed", doc_id=doc.doc_id, error=str(e))

        console.print(f"[green]✓ Created {len(all_chunks)} chunks[/green]")
        return all_chunks

    def build_indices(self, chunks: List[DocumentChunk]) -> None:
        """Build retrieval indices.

        Args:
            chunks: List of chunks
        """
        console.print("\n[cyan]Building indices...[/cyan]")

        try:
            with console.status("[yellow]Building FAISS and BM25 indices...[/yellow]"):
                self.retriever.build_index(chunks)

            console.print("[green]✓ Indices built successfully[/green]")

            # Save indices
            console.print("\n[cyan]Saving indices...[/cyan]")

            # Create output directory
            output_dir = Path("data/processed")
            output_dir.mkdir(parents=True, exist_ok=True)

            self.retriever.save_indices(
                dense_index_path="data/processed/faiss_index.bin",
                dense_chunks_path="data/processed/chunks.pkl",
                sparse_index_path="data/processed/bm25_index.pkl"
            )

            console.print("[green]✓ Indices saved[/green]")

        except Exception as e:
            console.print(f"[red]✗ Error building indices: {e}[/red]")
            logger.error("index_building_failed", error=str(e))
            raise

    def run_full_pipeline(self) -> None:
        """Run the complete processing pipeline."""
        console.print(Panel.fit(
            "[bold cyan]RAG Studienordnung Processing Pipeline[/bold cyan]\n"
            "Processing documents and building indices...",
            border_style="cyan"
        ))

        # Process sources
        documents = self.process_sources()

        if not documents:
            console.print("[red]No documents processed. Exiting.[/red]")
            return

        # Chunk documents
        chunks = self.chunk_documents(documents)

        if not chunks:
            console.print("[red]No chunks created. Exiting.[/red]")
            return

        # Build indices
        self.build_indices(chunks)

        console.print("\n[bold green]✓ Pipeline completed successfully![/bold green]")


@click.group()
def cli():
    """RAG Studienordnung processing pipeline."""
    pass


@cli.command()
@click.option('--config', default='config/config.yaml', help='Path to config file')
def build_index(config):
    """Build indices from configured sources."""
    pipeline = ProcessingPipeline(config)
    pipeline.run_full_pipeline()


@cli.command()
@click.option('--config', default='config/config.yaml', help='Path to config file')
def clear_cache(config):
    """Clear web scraping cache."""
    from src.scraping.cache_manager import CacheManager

    cache_config = load_config(config).get('scraping', {})
    cache_dir = cache_config.get('cache', {}).get('path', 'data/scraped')

    cache = CacheManager(cache_dir)
    cleared = cache.clear_all()

    console.print(f"[green]✓ Cleared {cleared} cache entries[/green]")


@cli.command()
@click.option('--config', default='config/config.yaml', help='Path to config file')
def stats(config):
    """Show cache statistics."""
    from src.scraping.cache_manager import CacheManager
    from rich.table import Table

    cache_config = load_config(config).get('scraping', {})
    cache_dir = cache_config.get('cache', {}).get('path', 'data/scraped')

    cache = CacheManager(cache_dir)
    stats = cache.get_stats()

    table = Table(title="Cache Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Entries", str(stats['total_entries']))
    table.add_row("Active Entries", str(stats['active_entries']))
    table.add_row("Expired Entries", str(stats['expired_entries']))
    table.add_row("Total Size (MB)", f"{stats['total_size_mb']:.2f}")

    console.print(table)


if __name__ == '__main__':
    # Import Panel here to avoid circular import
    from rich.panel import Panel
    cli()

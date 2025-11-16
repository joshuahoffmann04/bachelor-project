#!/usr/bin/env python3
"""Debug and diagnostic tool for RAG system."""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich import print as rprint

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger, get_logger
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.rag_generator import RAGGenerator
from src.preprocessing.document import DocumentChunk

console = Console()
logger = get_logger(__name__)


class RAGDebugger:
    """Debug and analyze RAG system performance."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize debugger."""
        self.config = load_config(config_path)
        setup_logger(log_level="DEBUG", log_format="console", enable_privacy=False)

        console.print("[yellow]Initializing debugger...[/yellow]")

        self.retriever = HybridRetriever(self.config)

        try:
            self.retriever.load_indices(
                dense_index_path="data/processed/faiss_index.bin",
                dense_chunks_path="data/processed/chunks.pkl",
                sparse_index_path="data/processed/bm25_index.pkl"
            )
            console.print("[green]✓ Indices loaded[/green]\n")
        except FileNotFoundError as e:
            console.print(f"[red]✗ Error loading indices: {e}[/red]")
            console.print("[yellow]Run: python pipeline.py build-index[/yellow]")
            sys.exit(1)

        self.generator = RAGGenerator(retriever=self.retriever, config=self.config)

    def analyze_chunks(self) -> None:
        """Analyze chunk statistics."""
        chunks = self.retriever.dense_retriever.chunks

        console.print(Panel.fit(
            "[bold cyan]Chunk Analysis[/bold cyan]",
            border_style="cyan"
        ))

        # Basic stats
        total_chunks = len(chunks)
        total_chars = sum(chunk.char_count for chunk in chunks)
        total_tokens = sum(chunk.tokens for chunk in chunks)

        avg_chars = total_chars / total_chunks if total_chunks > 0 else 0
        avg_tokens = total_tokens / total_chunks if total_chunks > 0 else 0

        stats_table = Table(title="Chunk Statistics", show_header=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total Chunks", str(total_chunks))
        stats_table.add_row("Total Characters", f"{total_chars:,}")
        stats_table.add_row("Total Tokens", f"{total_tokens:,}")
        stats_table.add_row("Avg Chars/Chunk", f"{avg_chars:.1f}")
        stats_table.add_row("Avg Tokens/Chunk", f"{avg_tokens:.1f}")

        console.print(stats_table)

        # Content type distribution
        from collections import Counter
        content_types = Counter(chunk.content_type.value for chunk in chunks)

        type_table = Table(title="\nContent Type Distribution", show_header=True)
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", style="green")
        type_table.add_column("Percentage", style="yellow")

        for content_type, count in content_types.most_common():
            percentage = (count / total_chunks) * 100
            type_table.add_row(content_type, str(count), f"{percentage:.1f}%")

        console.print(type_table)

        # Source distribution
        source_types = Counter(chunk.source_type.value for chunk in chunks)

        source_table = Table(title="\nSource Type Distribution", show_header=True)
        source_table.add_column("Source", style="cyan")
        source_table.add_column("Count", style="green")
        source_table.add_column("Percentage", style="yellow")

        for source_type, count in source_types.most_common():
            percentage = (count / total_chunks) * 100
            source_table.add_row(source_type, str(count), f"{percentage:.1f}%")

        console.print(source_table)

        # Show sample chunks
        console.print("\n[bold cyan]Sample Chunks:[/bold cyan]\n")

        for i, chunk in enumerate(chunks[:3], 1):
            console.print(f"[bold yellow]Chunk {i}:[/bold yellow]")
            console.print(f"  ID: {chunk.chunk_id}")
            console.print(f"  Type: {chunk.content_type.value}")
            console.print(f"  Source: {chunk.source_type.value}")
            console.print(f"  Tokens: {chunk.tokens}")
            console.print(f"  Content: {chunk.content[:200]}...")
            console.print(f"  Citation: {chunk.get_citation()}\n")

    async def test_retrieval(self, query: str, top_k: int = 10) -> None:
        """Test retrieval quality for a query."""
        console.print(Panel.fit(
            f"[bold cyan]Retrieval Analysis[/bold cyan]\nQuery: {query}",
            border_style="cyan"
        ))

        # Test dense retrieval
        console.print("\n[bold yellow]Dense Retrieval (FAISS):[/bold yellow]")
        dense_results = self.retriever.dense_retriever.search(query, top_k=top_k)

        dense_table = Table(show_header=True)
        dense_table.add_column("Rank", style="cyan", width=6)
        dense_table.add_column("Score", style="green", width=8)
        dense_table.add_column("Type", style="yellow", width=12)
        dense_table.add_column("Content Preview", style="white")

        for rank, (chunk, score) in enumerate(dense_results, 1):
            preview = chunk.content[:80].replace('\n', ' ')
            dense_table.add_row(
                str(rank),
                f"{score:.3f}",
                chunk.content_type.value,
                preview + "..."
            )

        console.print(dense_table)

        # Test sparse retrieval
        console.print("\n[bold yellow]Sparse Retrieval (BM25):[/bold yellow]")
        sparse_results = self.retriever.sparse_retriever.search(query, top_k=top_k)

        sparse_table = Table(show_header=True)
        sparse_table.add_column("Rank", style="cyan", width=6)
        sparse_table.add_column("Score", style="green", width=8)
        sparse_table.add_column("Type", style="yellow", width=12)
        sparse_table.add_column("Content Preview", style="white")

        for rank, (chunk, score) in enumerate(sparse_results, 1):
            preview = chunk.content[:80].replace('\n', ' ')
            sparse_table.add_row(
                str(rank),
                f"{score:.3f}",
                chunk.content_type.value,
                preview + "..."
            )

        console.print(sparse_table)

        # Test hybrid retrieval
        console.print("\n[bold yellow]Hybrid Retrieval (RRF Fusion):[/bold yellow]")
        hybrid_results = self.retriever.search(query, top_k=top_k)

        hybrid_table = Table(show_header=True)
        hybrid_table.add_column("Rank", style="cyan", width=6)
        hybrid_table.add_column("Score", style="green", width=8)
        hybrid_table.add_column("Type", style="yellow", width=12)
        hybrid_table.add_column("Source", style="magenta", width=15)
        hybrid_table.add_column("Content Preview", style="white")

        for rank, (chunk, score) in enumerate(hybrid_results, 1):
            preview = chunk.content[:60].replace('\n', ' ')
            source = (chunk.source_doc or "Unknown")[:15]
            hybrid_table.add_row(
                str(rank),
                f"{score:.3f}",
                chunk.content_type.value,
                source,
                preview + "..."
            )

        console.print(hybrid_table)

        # Show full content of top result
        if hybrid_results:
            console.print("\n[bold cyan]Top Result Full Content:[/bold cyan]\n")
            top_chunk, top_score = hybrid_results[0]
            console.print(Panel(
                f"{top_chunk.content}\n\n"
                f"[dim]Score: {top_score:.3f} | {top_chunk.get_citation()}[/dim]",
                title="Top Retrieved Chunk",
                border_style="green"
            ))

    async def test_generation(self, query: str) -> None:
        """Test full RAG generation."""
        console.print(Panel.fit(
            f"[bold cyan]Generation Test[/bold cyan]\nQuery: {query}",
            border_style="cyan"
        ))

        result = await self.generator.generate(query)

        # Show answer
        console.print("\n[bold green]Generated Answer:[/bold green]")
        console.print(Markdown(result['answer']))

        # Show metadata
        metadata = result.get('metadata', {})

        meta_table = Table(title="\nGeneration Metadata", show_header=True)
        meta_table.add_column("Metric", style="cyan")
        meta_table.add_column("Value", style="green")

        meta_table.add_row("Confidence", f"{result.get('confidence', 0):.3f}")
        meta_table.add_row("Num Chunks", str(metadata.get('num_chunks', 0)))
        meta_table.add_row("Retrieval Time", f"{metadata.get('retrieval_time_ms', 0):.0f}ms")
        meta_table.add_row("Generation Time", f"{metadata.get('generation_time_ms', 0):.0f}ms")
        meta_table.add_row("Total Time", f"{metadata.get('total_time_ms', 0):.0f}ms")

        if metadata.get('abstained'):
            meta_table.add_row("Status", "[yellow]ABSTAINED[/yellow]")

        console.print(meta_table)

        # Show sources
        if result.get('sources'):
            console.print("\n[bold blue]Sources:[/bold blue]")
            for idx, source in enumerate(result['sources'], 1):
                console.print(f"  [{idx}] {source}")

        # Show retrieved chunks
        if result.get('chunks'):
            console.print("\n[bold yellow]Retrieved Chunks:[/bold yellow]\n")
            for idx, chunk in enumerate(result['chunks'], 1):
                console.print(f"[bold]Chunk {idx}:[/bold]")
                console.print(f"  Type: {chunk.content_type.value}")
                console.print(f"  Score: {result['scores'][idx-1]:.3f}")
                console.print(f"  Content: {chunk.content[:150]}...\n")

    async def run_test_suite(self) -> None:
        """Run a suite of test queries."""
        test_queries = [
            # ECTS lookups
            "Wie viele ECTS hat das Modul Algorithmen und Datenstrukturen?",
            "ECTS Datenbanken",

            # Prerequisites
            "Welche Voraussetzungen gibt es für die Bachelorarbeit?",
            "Voraussetzungen Softwaretechnik",

            # Deadlines
            "Bis wann muss ich mich für Prüfungen anmelden?",
            "Anmeldefrist Prüfung",

            # Exam formats
            "Welche Prüfungsform hat das Modul Programmierung?",

            # Out of scope (should abstain)
            "Wie ist das Wetter heute?",
            "Was soll ich zum Mittagessen essen?",
        ]

        results = []

        console.print(Panel.fit(
            "[bold cyan]Running Test Suite[/bold cyan]\n"
            f"Testing {len(test_queries)} queries...",
            border_style="cyan"
        ))

        for idx, query in enumerate(test_queries, 1):
            console.print(f"\n[bold yellow]Test {idx}/{len(test_queries)}:[/bold yellow] {query}")

            result = await self.generator.generate(query)

            # Simplified output
            confidence = result.get('confidence', 0)
            abstained = result.get('metadata', {}).get('abstained', False)

            status = "✓" if confidence > 0.6 else "✗"
            if abstained:
                status = "⊘"

            console.print(f"  {status} Confidence: {confidence:.2f}")
            console.print(f"  Answer: {result['answer'][:100]}...")

            results.append({
                'query': query,
                'confidence': confidence,
                'abstained': abstained,
                'answer_length': len(result['answer'])
            })

        # Summary
        console.print("\n" + "="*80)
        console.print("[bold cyan]Test Summary:[/bold cyan]\n")

        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        num_abstained = sum(1 for r in results if r['abstained'])

        summary_table = Table(show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Tests", str(len(results)))
        summary_table.add_row("Avg Confidence", f"{avg_confidence:.3f}")
        summary_table.add_row("Abstained", f"{num_abstained}/{len(results)}")
        summary_table.add_row(
            "High Confidence (>0.6)",
            str(sum(1 for r in results if r['confidence'] > 0.6))
        )

        console.print(summary_table)


@click.group()
def cli():
    """RAG debugging and diagnostic tools."""
    pass


@cli.command()
@click.option('--config', default='config/config.yaml', help='Config file path')
def analyze_chunks(config):
    """Analyze chunk statistics and distribution."""
    debugger = RAGDebugger(config)
    debugger.analyze_chunks()


@cli.command()
@click.argument('query')
@click.option('--top-k', default=10, help='Number of results to retrieve')
@click.option('--config', default='config/config.yaml', help='Config file path')
def test_retrieval(query, top_k, config):
    """Test retrieval quality for a specific query."""
    debugger = RAGDebugger(config)
    asyncio.run(debugger.test_retrieval(query, top_k))


@cli.command()
@click.argument('query')
@click.option('--config', default='config/config.yaml', help='Config file path')
def test_generation(query, config):
    """Test full RAG generation for a query."""
    debugger = RAGDebugger(config)
    asyncio.run(debugger.test_generation(query))


@cli.command()
@click.option('--config', default='config/config.yaml', help='Config file path')
def test_suite(config):
    """Run full test suite with predefined queries."""
    debugger = RAGDebugger(config)
    asyncio.run(debugger.run_test_suite())


@cli.command()
@click.option('--config', default='config/config.yaml', help='Config file path')
def full_report(config):
    """Generate comprehensive diagnostic report."""
    debugger = RAGDebugger(config)

    console.print("[bold cyan]Full Diagnostic Report[/bold cyan]\n")

    debugger.analyze_chunks()

    console.print("\n" + "="*80 + "\n")

    asyncio.run(debugger.run_test_suite())


if __name__ == '__main__':
    cli()

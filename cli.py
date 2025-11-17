#!/usr/bin/env python3
"""Comprehensive CLI for RAG Studienordnung system.

This CLI provides multiple modes of interaction:
- Interactive mode for conversational queries
- Single-question mode for scripting
- Index building and management
- Statistics and monitoring
- Evaluation (when test datasets are available)
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger, get_logger
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.rag_generator import RAGGenerator
from src.preprocessing.document import DocumentChunk

console = Console()
logger = get_logger(__name__)


class InteractiveSession:
    """Interactive CLI session with conversation history."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize interactive session.

        Args:
            config_path: Path to configuration file
        """
        console.print("[yellow]Loading configuration and models...[/yellow]")
        self.config = load_config(config_path)

        # Setup logging
        log_config = self.config.get('logging', {})
        setup_logger(
            log_level=log_config.get('level', 'INFO'),
            log_format='console',
            enable_privacy=log_config.get('privacy', {}).get('anonymize_queries', False)
        )

        # Load retriever
        console.print("[yellow]Loading retrieval indices...[/yellow]")
        self.retriever = HybridRetriever(self.config)

        try:
            self.retriever.load_indices(
                dense_index_path="data/processed/faiss_index.bin",
                dense_chunks_path="data/processed/chunks.pkl",
                sparse_index_path="data/processed/bm25_index.pkl"
            )
            console.print("[green]✓ Indices loaded successfully[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to load indices: {e}[/red]")
            console.print("[yellow]Please run 'python pipeline.py build-index' first[/yellow]")
            sys.exit(1)

        # Initialize RAG generator
        console.print("[yellow]Initializing LLM...[/yellow]")
        self.generator = RAGGenerator(self.retriever, config=self.config)
        console.print("[green]✓ System ready[/green]\n")

        # Session state
        self.conversation_history = []
        self.total_queries = 0

    def show_help(self):
        """Display help message."""
        help_text = """
        ## Available Commands

        - `/help` - Show this help message
        - `/stats` - Show session statistics
        - `/reset` - Reset conversation history
        - `/quit` or `/exit` - Exit the session

        ## Tips

        - Ask specific questions about study regulations, modules, ECTS, deadlines, etc.
        - The system will cite sources for all answers
        - If confidence is low, the system will abstain from answering
        """
        console.print(Panel(Markdown(help_text), title="Help", border_style="cyan"))

    def show_stats(self):
        """Display session statistics."""
        table = Table(title="Session Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Queries", str(self.total_queries))
        table.add_row("Conversation History", str(len(self.conversation_history)))

        # Add retriever stats if available
        if hasattr(self.retriever, 'dense_retriever') and self.retriever.dense_retriever:
            num_chunks = len(self.retriever.dense_retriever.chunks)
            table.add_row("Indexed Chunks", str(num_chunks))

        console.print(table)

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        console.print("[green]✓ Conversation history reset[/green]")

    async def ask_question(self, query: str, show_metadata: bool = False):
        """Ask a question and display the answer.

        Args:
            query: User question
            show_metadata: Whether to show detailed metadata
        """
        self.total_queries += 1

        # Generate answer
        with console.status("[yellow]Generating answer...[/yellow]"):
            result = await self.generator.generate(query)

        # Display answer
        console.print("\n[bold cyan]Answer:[/bold cyan]")
        console.print(Panel(result['answer'], border_style="green"))

        # Display sources
        if result.get('sources'):
            console.print("\n[bold cyan]Sources:[/bold cyan]")
            for idx, source in enumerate(result['sources'], 1):
                console.print(f"  [{idx}] {source}")

        # Display metadata if requested
        if show_metadata and result.get('metadata'):
            console.print("\n[bold cyan]Metadata:[/bold cyan]")
            metadata = result['metadata']
            console.print(f"  Retrieval time: {metadata.get('retrieval_time_ms', 0):.1f} ms")
            console.print(f"  Generation time: {metadata.get('generation_time_ms', 0):.1f} ms")
            console.print(f"  Total time: {metadata.get('total_time_ms', 0):.1f} ms")
            console.print(f"  Confidence: {result.get('confidence', 0):.3f}")

            if metadata.get('abstained'):
                console.print("  [yellow]⚠ System abstained (low confidence)[/yellow]")

        console.print()  # Empty line for spacing

        # Add to conversation history
        self.conversation_history.append({
            'query': query,
            'answer': result['answer'],
            'confidence': result.get('confidence', 0)
        })

    async def run(self, show_metadata: bool = False):
        """Run interactive session.

        Args:
            show_metadata: Whether to show metadata for each answer
        """
        # Show welcome message
        console.print(Panel.fit(
            "[bold cyan]RAG Studienordnung - Interactive Mode[/bold cyan]\n"
            "Ask questions about study regulations, modules, and requirements.\n"
            "Type /help for commands, /quit to exit.",
            border_style="cyan"
        ))

        while True:
            try:
                # Get user input
                query = Prompt.ask("\n[bold green]Your question[/bold green]")

                # Handle commands
                if query.startswith('/'):
                    command = query.lower().strip()

                    if command in ['/quit', '/exit']:
                        console.print("[yellow]Goodbye![/yellow]")
                        break

                    elif command == '/help':
                        self.show_help()

                    elif command == '/stats':
                        self.show_stats()

                    elif command == '/reset':
                        self.reset_conversation()

                    else:
                        console.print(f"[red]Unknown command: {command}[/red]")
                        console.print("[yellow]Type /help for available commands[/yellow]")

                    continue

                # Skip empty queries
                if not query.strip():
                    continue

                # Process question
                await self.ask_question(query, show_metadata)

            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Goodbye![/yellow]")
                break

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                logger.error("interactive_error", error=str(e))


@click.group()
def cli():
    """RAG Studienordnung - Retrieval Augmented Generation for Study Regulations."""
    pass


@cli.command()
@click.option('--config', default='config/config.yaml', help='Path to config file')
@click.option('--metadata/--no-metadata', default=False, help='Show metadata for answers')
def interactive(config, metadata):
    """Start interactive question-answering session."""
    session = InteractiveSession(config)
    asyncio.run(session.run(show_metadata=metadata))


@cli.command()
@click.argument('question')
@click.option('--config', default='config/config.yaml', help='Path to config file')
@click.option('--metadata/--no-metadata', default=False, help='Show metadata')
@click.option('--json/--no-json', default=False, help='Output as JSON')
def ask(question, config, metadata, json):
    """Ask a single question (for scripting/automation).

    QUESTION: The question to ask
    """
    import json as json_module

    # Load config and initialize
    config_dict = load_config(config)
    setup_logger(
        log_level=config_dict.get('logging', {}).get('level', 'INFO'),
        log_format='console',
        enable_privacy=False
    )

    # Load retriever
    retriever = HybridRetriever(config_dict)
    try:
        retriever.load_indices(
            dense_index_path="data/processed/faiss_index.bin",
            dense_chunks_path="data/processed/chunks.pkl",
            sparse_index_path="data/processed/bm25_index.pkl"
        )
    except Exception as e:
        console.print(f"[red]✗ Failed to load indices: {e}[/red]")
        console.print("[yellow]Please run 'python pipeline.py build-index' first[/yellow]")
        sys.exit(1)

    # Generate answer
    generator = RAGGenerator(retriever, config=config_dict)

    async def generate():
        return await generator.generate(question)

    result = asyncio.run(generate())

    # Output
    if json:
        # JSON output for scripting
        output = {
            'question': question,
            'answer': result['answer'],
            'sources': result.get('sources', []),
            'confidence': result.get('confidence', 0),
            'metadata': result.get('metadata', {})
        }
        console.print(json_module.dumps(output, ensure_ascii=False, indent=2))
    else:
        # Human-readable output
        console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")
        console.print("[bold cyan]Answer:[/bold cyan]")
        console.print(Panel(result['answer'], border_style="green"))

        if result.get('sources'):
            console.print("\n[bold cyan]Sources:[/bold cyan]")
            for idx, source in enumerate(result['sources'], 1):
                console.print(f"  [{idx}] {source}")

        if metadata and result.get('metadata'):
            console.print(f"\n[bold cyan]Confidence:[/bold cyan] {result.get('confidence', 0):.3f}")
            metadata_dict = result['metadata']
            console.print(f"[bold cyan]Response time:[/bold cyan] {metadata_dict.get('total_time_ms', 0):.0f} ms")


@cli.command()
@click.option('--config', default='config/config.yaml', help='Path to config file')
def stats(config):
    """Show system statistics and index information."""
    from src.scraping.cache_manager import CacheManager

    config_dict = load_config(config)

    # Index stats
    console.print("[bold cyan]Index Statistics[/bold cyan]\n")

    try:
        retriever = HybridRetriever(config_dict)
        retriever.load_indices(
            dense_index_path="data/processed/faiss_index.bin",
            dense_chunks_path="data/processed/chunks.pkl",
            sparse_index_path="data/processed/bm25_index.pkl"
        )

        table = Table(show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        if hasattr(retriever, 'dense_retriever') and retriever.dense_retriever:
            num_chunks = len(retriever.dense_retriever.chunks)
            table.add_row("Total Chunks", str(num_chunks))

            # Calculate total content size
            total_chars = sum(len(chunk.content) for chunk in retriever.dense_retriever.chunks)
            table.add_row("Total Content Size", f"{total_chars:,} characters")

            # Count source documents
            sources = set(chunk.source_doc for chunk in retriever.dense_retriever.chunks if chunk.source_doc)
            table.add_row("Source Documents", str(len(sources)))

        console.print(table)

    except FileNotFoundError:
        console.print("[yellow]No indices found. Run 'python pipeline.py build-index' first.[/yellow]")

    # Cache stats
    console.print("\n[bold cyan]Cache Statistics[/bold cyan]\n")

    cache_config = config_dict.get('scraping', {})
    cache_dir = cache_config.get('cache', {}).get('path', 'data/scraped')

    if Path(cache_dir).exists():
        cache = CacheManager(cache_dir)
        cache_stats = cache.get_stats()

        cache_table = Table(show_header=True)
        cache_table.add_column("Metric", style="cyan")
        cache_table.add_column("Value", style="green")

        cache_table.add_row("Total Entries", str(cache_stats['total_entries']))
        cache_table.add_row("Active Entries", str(cache_stats['active_entries']))
        cache_table.add_row("Expired Entries", str(cache_stats['expired_entries']))
        cache_table.add_row("Total Size", f"{cache_stats['total_size_mb']:.2f} MB")

        console.print(cache_table)
    else:
        console.print("[yellow]No cache directory found.[/yellow]")

    # Device info
    console.print("\n[bold cyan]Device Configuration[/bold cyan]\n")
    try:
        from src.utils.device import format_device_info
        console.print(format_device_info())
    except Exception as e:
        console.print(f"[yellow]Could not get device info: {e}[/yellow]")


@cli.command()
@click.option('--config', default='config/config.yaml', help='Path to config file')
def build_index(config):
    """Build indices from configured sources (alias for pipeline.py build-index)."""
    from pipeline import ProcessingPipeline

    pipeline = ProcessingPipeline(config)
    pipeline.run_full_pipeline()


@cli.command()
@click.option('--config', default='config/config.yaml', help='Path to config file')
@click.option('--test-set', default='data/test_sets/', help='Path to test dataset')
@click.option('--output', default='evaluation_results.json', help='Output file')
@click.option('--verbose', is_flag=True, help='Show detailed results')
def evaluate(config, test_set, output, verbose):
    """Run evaluation on test dataset.

    Evaluates the RAG system using:
    - RAGAS metrics (context relevance, faithfulness, answer relevance)
    - Custom metrics (ECTS accuracy, reference quality, abstaining rate, hallucination detection)
    - Exports results to JSON and Markdown formats
    """
    from pipeline import ProcessingPipeline

    pipeline = ProcessingPipeline(config)
    pipeline.evaluate(test_set, output, verbose)


if __name__ == '__main__':
    cli()

#!/usr/bin/env python3
"""Interactive CLI for RAG Studienordnung system."""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger, get_logger
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.rag_generator import RAGGenerator

console = Console()
logger = get_logger(__name__)


class RAGChatbot:
    """Interactive RAG chatbot."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize chatbot.

        Args:
            config_path: Path to configuration file
        """
        # Load config
        console.print("[yellow]Loading configuration...[/yellow]")
        self.config = load_config(config_path)

        # Setup logging
        log_config = self.config.get('logging', {})
        setup_logger(
            log_level=log_config.get('level', 'INFO'),
            log_format=log_config.get('format', 'console'),
            log_file=log_config.get('file', {}).get('path') if log_config.get('file', {}).get('enabled') else None,
            enable_privacy=log_config.get('privacy', {}).get('anonymize_queries', True)
        )

        # Initialize retriever
        console.print("[yellow]Initializing retriever...[/yellow]")
        self.retriever = HybridRetriever(self.config)

        # Load indices
        self._load_indices()

        # Initialize generator
        console.print("[yellow]Initializing LLM...[/yellow]")
        self.generator = RAGGenerator(
            retriever=self.retriever,
            config=self.config
        )

        console.print("[green]✓ System ready![/green]\n")

    def _load_indices(self):
        """Load pre-built indices."""
        try:
            self.retriever.load_indices(
                dense_index_path="data/processed/faiss_index.bin",
                dense_chunks_path="data/processed/chunks.pkl",
                sparse_index_path="data/processed/bm25_index.pkl"
            )
            console.print("[green]✓ Indices loaded successfully[/green]")
        except FileNotFoundError as e:
            console.print(
                f"[red]✗ Indices not found: {e}[/red]\n"
                "[yellow]Please run the indexing pipeline first:[/yellow]\n"
                "  python pipeline.py build-index"
            )
            sys.exit(1)

    async def ask(self, question: str) -> None:
        """Ask a question and display answer.

        Args:
            question: User question
        """
        console.print(Panel(f"[bold cyan]Frage:[/bold cyan] {question}", expand=False))

        with console.status("[yellow]Suche nach relevanten Informationen...[/yellow]"):
            result = await self.generator.generate(question)

        # Display answer
        answer = result['answer']
        console.print("\n[bold green]Antwort:[/bold green]")
        console.print(Markdown(answer))

        # Display sources
        if result['sources']:
            console.print("\n[bold blue]Quellen:[/bold blue]")
            for idx, source in enumerate(result['sources'], 1):
                console.print(f"  [{idx}] {source}")

        # Display metadata
        metadata = result.get('metadata', {})
        console.print(f"\n[dim]Konfidenz: {result.get('confidence', 0):.2f}[/dim]")
        console.print(
            f"[dim]Zeit: Retrieval {metadata.get('retrieval_time_ms', 0):.0f}ms, "
            f"Generation {metadata.get('generation_time_ms', 0):.0f}ms[/dim]\n"
        )

    async def interactive_mode(self):
        """Run interactive chat mode."""
        console.print(Panel.fit(
            "[bold cyan]RAG Studienordnung Chatbot[/bold cyan]\n"
            "Stellen Sie Fragen zu Studienordnungen, Modulhandbüchern und Prüfungsregelungen.\n"
            "Befehle: /quit, /help, /stats",
            border_style="cyan"
        ))

        while True:
            try:
                # Get user input
                question = console.input("\n[bold yellow]>[/bold yellow] ")

                if not question.strip():
                    continue

                # Handle commands
                if question.startswith('/'):
                    if question == '/quit' or question == '/exit':
                        console.print("[yellow]Auf Wiedersehen![/yellow]")
                        break
                    elif question == '/help':
                        self._show_help()
                    elif question == '/stats':
                        self._show_stats()
                    else:
                        console.print(f"[red]Unbekannter Befehl: {question}[/red]")
                    continue

                # Ask question
                await self.ask(question)

            except KeyboardInterrupt:
                console.print("\n[yellow]Unterbrochen. /quit zum Beenden.[/yellow]")
            except Exception as e:
                console.print(f"[red]Fehler: {e}[/red]")
                logger.error("interactive_error", error=str(e))

    def _show_help(self):
        """Show help message."""
        help_text = """
        [bold cyan]Verfügbare Befehle:[/bold cyan]

        /help  - Zeige diese Hilfe
        /stats - Zeige System-Statistiken
        /quit  - Beende das Programm

        [bold cyan]Beispielfragen:[/bold cyan]

        - Wie viele ECTS hat das Modul "Algorithmen und Datenstrukturen"?
        - Welche Voraussetzungen gibt es für die Bachelorarbeit?
        - Bis wann muss ich mich für Prüfungen anmelden?
        - Welche Prüfungsform hat das Modul Datenbanken?
        """
        console.print(Panel(help_text, border_style="cyan"))

    def _show_stats(self):
        """Show system statistics."""
        stats = self.retriever.get_stats()

        table = Table(title="System Statistiken", show_header=True)
        table.add_column("Komponente", style="cyan")
        table.add_column("Metrik", style="yellow")
        table.add_column("Wert", style="green")

        # Dense retriever stats
        dense_stats = stats.get('dense', {})
        table.add_row("Dense Retrieval", "Chunks", str(dense_stats.get('num_chunks', 0)))
        table.add_row("", "Embedding Dim", str(dense_stats.get('embedding_dim', 0)))
        table.add_row("", "Model", dense_stats.get('model_name', 'N/A'))

        # Sparse retriever stats
        sparse_stats = stats.get('sparse', {})
        table.add_row("Sparse Retrieval", "Chunks", str(sparse_stats.get('num_chunks', 0)))
        table.add_row("", "Avg Doc Length", f"{sparse_stats.get('avg_doc_length', 0):.1f}")

        # Fusion stats
        fusion_stats = stats.get('fusion', {})
        table.add_row("Fusion", "Method", fusion_stats.get('method', 'N/A'))
        table.add_row("", "Dense Weight", f"{fusion_stats.get('dense_weight', 0):.2f}")
        table.add_row("", "Sparse Weight", f"{fusion_stats.get('sparse_weight', 0):.2f}")

        console.print(table)


@click.group()
def cli():
    """RAG Studienordnung CLI."""
    pass


@cli.command()
@click.option('--config', default='config/config.yaml', help='Path to config file')
def interactive(config):
    """Start interactive chat mode."""
    bot = RAGChatbot(config)
    asyncio.run(bot.interactive_mode())


@cli.command()
@click.argument('question')
@click.option('--config', default='config/config.yaml', help='Path to config file')
def ask(question, config):
    """Ask a single question."""
    bot = RAGChatbot(config)
    asyncio.run(bot.ask(question))


if __name__ == '__main__':
    cli()

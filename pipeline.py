#!/usr/bin/env python3
"""Main pipeline for document processing and indexing."""

import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any

import click
from rich.console import Console
from rich.progress import track
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger, get_logger
from src.preprocessing.pdf_parser import PDFParser
from src.preprocessing.document import Document, DocumentChunk
from src.scraping.web_scraper import WebScraper
from src.chunking.chunker import HybridChunker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.rag_generator import RAGGenerator
from src.evaluation.evaluator import RAGASEvaluator
from src.evaluation.metrics import CustomMetrics

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

    def evaluate(
        self,
        test_set_path: str,
        output_path: str = "evaluation_results.json",
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Run evaluation on test dataset.

        Args:
            test_set_path: Path to test dataset (JSONL file or directory with JSONL files)
            output_path: Path for output file
            verbose: Show detailed results

        Returns:
            Dictionary with aggregated results
        """
        from rich.panel import Panel

        console.print(Panel.fit(
            "[bold cyan]RAG System Evaluation[/bold cyan]\n"
            f"Test Set: {test_set_path}",
            border_style="cyan"
        ))

        # Load retriever
        console.print("[yellow]Loading retrieval indices...[/yellow]")
        try:
            self.retriever.load_indices(
                dense_index_path="data/processed/faiss_index.bin",
                dense_chunks_path="data/processed/chunks.pkl",
                sparse_index_path="data/processed/bm25_index.pkl"
            )
            console.print("[green]✓ Indices loaded[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to load indices: {e}[/red]")
            console.print("[yellow]Please run 'python pipeline.py build-index' first[/yellow]")
            return {}

        # Initialize RAG generator
        console.print("[yellow]Initializing RAG generator...[/yellow]")
        generator = RAGGenerator(self.retriever, config=self.config)
        console.print("[green]✓ Generator ready[/green]\n")

        # Load test cases
        test_cases = self._load_test_cases(test_set_path)
        if not test_cases:
            console.print("[red]No test cases found[/red]")
            return {}

        console.print(f"[cyan]Found {len(test_cases)} test cases[/cyan]\n")

        # Generate answers for all test cases
        console.print("[yellow]Generating answers...[/yellow]")
        test_data = []

        async def generate_all():
            results = []
            for test_case in track(test_cases, description="Evaluating..."):
                result = await generator.generate(test_case['question'])
                results.append({
                    'question': test_case['question'],
                    'answer': result['answer'],
                    'contexts': [chunk.content for chunk in result.get('chunks', [])],
                    'ground_truth': test_case.get('ground_truth_answer'),
                    'metadata': {
                        **test_case.get('metadata', {}),
                        'category': test_case.get('category'),
                        'difficulty': test_case.get('difficulty'),
                        'ects_value': test_case.get('ects_value'),
                        'confidence': result.get('confidence', 0),
                        'retrieval_time_ms': result.get('metadata', {}).get('retrieval_time_ms', 0),
                        'generation_time_ms': result.get('metadata', {}).get('generation_time_ms', 0)
                    }
                })
            return results

        test_data = asyncio.run(generate_all())
        console.print(f"[green]✓ Generated {len(test_data)} answers[/green]\n")

        # Compute custom metrics
        def compute_custom_metrics(test_case: Dict[str, Any]) -> Dict[str, float]:
            """Compute custom metrics for a test case."""
            metrics = {}

            # ECTS accuracy (if ground truth ECTS is available)
            if test_case.get('metadata', {}).get('ects_value'):
                ects_result = CustomMetrics.ects_accuracy(
                    test_case['answer'],
                    str(test_case['metadata']['ects_value']) + " ECTS",
                    tolerance=0
                )
                metrics['ects_accuracy'] = ects_result.score

            # Reference quality
            ref_result = CustomMetrics.reference_quality(
                test_case['answer'],
                test_case.get('contexts', []),
                required_citations=1
            )
            metrics['reference_quality'] = ref_result.score

            # Hallucination detection
            hal_result = CustomMetrics.hallucination_detection(
                test_case['answer'],
                test_case.get('contexts', [])
            )
            metrics['hallucination_score'] = hal_result.score

            return metrics

        # Run evaluation
        console.print("[yellow]Running RAGAS evaluation...[/yellow]")
        evaluator = RAGASEvaluator(self.config)
        results = evaluator.evaluate_batch(test_data, custom_metric_fn=compute_custom_metrics)

        # Compute abstaining rate across all answers
        all_answers = [tc['answer'] for tc in test_data]
        abstaining_result = CustomMetrics.abstaining_rate(all_answers)

        # Add abstaining rate to aggregated metrics
        results.metrics['custom_abstaining_rate'] = {
            'mean': abstaining_result.score,
            'median': abstaining_result.score,
            'std': 0.0,
            'min': abstaining_result.score,
            'max': abstaining_result.score,
            'count': len(all_answers)
        }

        console.print("[green]✓ Evaluation complete[/green]\n")

        # Display results
        self._display_results(results, verbose)

        # Export results
        console.print(f"\n[yellow]Exporting results to {output_path}...[/yellow]")
        evaluator.export_results(results, output_path, format='json')

        # Also export markdown report
        md_path = output_path.replace('.json', '.md')
        evaluator.export_results(results, md_path, format='markdown')

        console.print(f"[green]✓ Results exported to:[/green]")
        console.print(f"  - JSON: {output_path}")
        console.print(f"  - Markdown: {md_path}")

        return results.to_dict()

    def _load_test_cases(self, path: str) -> List[Dict[str, Any]]:
        """Load test cases from JSONL file or directory.

        Args:
            path: Path to JSONL file or directory

        Returns:
            List of test cases
        """
        test_cases = []
        path_obj = Path(path)

        if path_obj.is_file():
            # Single file
            test_cases.extend(self._load_jsonl(path_obj))
        elif path_obj.is_dir():
            # Directory - load all JSONL files
            for jsonl_file in path_obj.glob("*.jsonl"):
                test_cases.extend(self._load_jsonl(jsonl_file))
        else:
            console.print(f"[red]Invalid path: {path}[/red]")

        return test_cases

    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load test cases from a single JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of test cases
        """
        test_cases = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        test_case = json.loads(line)
                        test_cases.append(test_case)
                    except json.JSONDecodeError as e:
                        logger.error("invalid_json", file=str(file_path), line=line_num, error=str(e))

        except FileNotFoundError:
            console.print(f"[red]File not found: {file_path}[/red]")

        return test_cases

    def _display_results(self, results, verbose: bool = False):
        """Display evaluation results in a table.

        Args:
            results: AggregatedResults object
            verbose: Show detailed individual results
        """
        from rich.panel import Panel

        # Summary table
        table = Table(title="Evaluation Metrics Summary", show_header=True)
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Mean", style="green", justify="right")
        table.add_column("Median", style="green", justify="right")
        table.add_column("Std", style="yellow", justify="right")
        table.add_column("Min", style="red", justify="right")
        table.add_column("Max", style="green", justify="right")

        for metric_name, stats in results.metrics.items():
            table.add_row(
                metric_name,
                f"{stats['mean']:.3f}",
                f"{stats['median']:.3f}",
                f"{stats['std']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}"
            )

        console.print(table)

        # Category breakdown if available
        categories = {}
        for result in results.results:
            category = result.metadata.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        if len(categories) > 1:
            console.print("\n[bold cyan]Results by Category:[/bold cyan]")
            cat_table = Table(show_header=True)
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Count", style="green", justify="right")
            cat_table.add_column("Avg Confidence", style="yellow", justify="right")

            for cat, cat_results in categories.items():
                avg_confidence = sum(
                    r.metadata.get('confidence', 0) for r in cat_results
                ) / len(cat_results)

                cat_table.add_row(
                    cat,
                    str(len(cat_results)),
                    f"{avg_confidence:.3f}"
                )

            console.print(cat_table)

        # Verbose output
        if verbose:
            console.print("\n[bold cyan]Individual Results:[/bold cyan]")
            for idx, result in enumerate(results.results[:10], 1):  # Show first 10
                console.print(f"\n[bold]Query {idx}:[/bold] {result.question[:100]}...")
                console.print(f"[dim]Answer:[/dim] {result.answer[:200]}...")

                if result.context_relevance is not None:
                    console.print(f"  Context Relevance: {result.context_relevance:.3f}")
                if result.faithfulness is not None:
                    console.print(f"  Faithfulness: {result.faithfulness:.3f}")
                if result.answer_relevance is not None:
                    console.print(f"  Answer Relevance: {result.answer_relevance:.3f}")

                for metric, score in result.custom_metrics.items():
                    console.print(f"  {metric}: {score:.3f}")


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

    cache_config = load_config(config).get('scraping', {})
    cache_dir = cache_config.get('cache', {}).get('path', 'data/scraped')

    cache = CacheManager(cache_dir)
    stats_data = cache.get_stats()

    table = Table(title="Cache Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Entries", str(stats_data['total_entries']))
    table.add_row("Active Entries", str(stats_data['active_entries']))
    table.add_row("Expired Entries", str(stats_data['expired_entries']))
    table.add_row("Total Size (MB)", f"{stats_data['total_size_mb']:.2f}")

    console.print(table)


@cli.command()
@click.option('--config', default='config/config.yaml', help='Path to config file')
@click.option('--test-set', default='data/test_sets/', help='Path to test dataset')
@click.option('--output', default='evaluation_results.json', help='Output file')
@click.option('--verbose', is_flag=True, help='Show detailed results')
def evaluate(config, test_set, output, verbose):
    """Run evaluation on test dataset."""
    pipeline = ProcessingPipeline(config)
    pipeline.evaluate(test_set, output, verbose)


if __name__ == '__main__':
    # Import Panel here to avoid circular import
    from rich.panel import Panel
    cli()

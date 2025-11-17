#!/usr/bin/env python3
"""RAGAS-based evaluator for RAG system."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single query.

    Attributes:
        question: The question asked
        answer: The generated answer
        ground_truth: Expected/reference answer (if available)
        retrieved_contexts: List of retrieved text chunks
        context_relevance: Score for context relevance (0-1)
        faithfulness: Score for answer faithfulness to context (0-1)
        answer_relevance: Score for answer relevance to question (0-1)
        answer_correctness: Score for correctness vs ground truth (0-1, if ground truth available)
        custom_metrics: Dictionary of custom metric scores
        metadata: Additional metadata
    """
    question: str
    answer: str
    ground_truth: Optional[str] = None
    retrieved_contexts: List[str] = None
    context_relevance: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    answer_correctness: Optional[float] = None
    custom_metrics: Dict[str, float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.retrieved_contexts is None:
            self.retrieved_contexts = []
        if self.custom_metrics is None:
            self.custom_metrics = {}
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AggregatedResults:
    """Aggregated evaluation results across multiple queries.

    Attributes:
        total_queries: Total number of evaluated queries
        metrics: Dictionary of metric name to aggregated scores
        results: List of individual evaluation results
        timestamp: When the evaluation was run
        config: Configuration used for evaluation
    """
    total_queries: int
    metrics: Dict[str, Dict[str, float]]  # metric_name -> {mean, median, std, min, max}
    results: List[EvaluationResult]
    timestamp: str = None
    config: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize timestamp."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.config is None:
            self.config = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_queries': self.total_queries,
            'metrics': self.metrics,
            'results': [r.to_dict() for r in self.results],
            'timestamp': self.timestamp,
            'config': self.config
        }


class RAGASEvaluator:
    """Evaluator using RAGAS metrics for RAG system evaluation.

    This class integrates RAGAS (Retrieval Augmented Generation Assessment)
    metrics to evaluate the quality of the RAG system.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RAGAS evaluator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.eval_config = self.config.get('evaluation', {})

        # RAGAS configuration
        self.ragas_config = self.eval_config.get('ragas', {})
        self.enabled_metrics = self.ragas_config.get('metrics', [
            'context_relevance',
            'faithfulness',
            'answer_relevance',
            'answer_correctness'
        ])

        # Check if RAGAS is available
        self.ragas_available = self._check_ragas_availability()

        logger.info("ragas_evaluator_initialized",
                    ragas_available=self.ragas_available,
                    metrics=self.enabled_metrics)

    def _check_ragas_availability(self) -> bool:
        """Check if RAGAS library is available.

        Returns:
            True if RAGAS can be imported, False otherwise
        """
        try:
            import ragas
            return True
        except ImportError:
            logger.warning("ragas_not_available",
                          message="RAGAS library not available. Install with: pip install ragas")
            return False

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        custom_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluate a single question-answer pair.

        Args:
            question: The question asked
            answer: The generated answer
            contexts: Retrieved context chunks
            ground_truth: Expected answer (optional)
            custom_metrics: Pre-computed custom metrics
            metadata: Additional metadata

        Returns:
            EvaluationResult with all metrics
        """
        result = EvaluationResult(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            retrieved_contexts=contexts,
            custom_metrics=custom_metrics or {},
            metadata=metadata or {}
        )

        # If RAGAS is not available, return result with only custom metrics
        if not self.ragas_available:
            logger.warning("ragas_evaluation_skipped",
                          reason="RAGAS library not available")
            return result

        try:
            # Import RAGAS metrics
            from ragas import evaluate
            from ragas.metrics import (
                context_relevancy,
                faithfulness,
                answer_relevancy,
                answer_correctness
            )
            from datasets import Dataset

            # Prepare data for RAGAS
            data = {
                'question': [question],
                'answer': [answer],
                'contexts': [contexts],
            }

            if ground_truth:
                data['ground_truth'] = [ground_truth]

            dataset = Dataset.from_dict(data)

            # Select metrics based on configuration
            metrics_to_run = []

            if 'context_relevance' in self.enabled_metrics:
                metrics_to_run.append(context_relevancy)

            if 'faithfulness' in self.enabled_metrics:
                metrics_to_run.append(faithfulness)

            if 'answer_relevance' in self.enabled_metrics:
                metrics_to_run.append(answer_relevancy)

            if 'answer_correctness' in self.enabled_metrics and ground_truth:
                metrics_to_run.append(answer_correctness)

            # Run evaluation
            if metrics_to_run:
                eval_result = evaluate(dataset, metrics=metrics_to_run)

                # Extract scores
                if 'context_relevancy' in eval_result:
                    result.context_relevance = eval_result['context_relevancy']

                if 'faithfulness' in eval_result:
                    result.faithfulness = eval_result['faithfulness']

                if 'answer_relevancy' in eval_result:
                    result.answer_relevance = eval_result['answer_relevancy']

                if 'answer_correctness' in eval_result:
                    result.answer_correctness = eval_result['answer_correctness']

        except Exception as e:
            logger.error("ragas_evaluation_failed",
                        error=str(e),
                        question=question[:100])

        return result

    def evaluate_batch(
        self,
        test_data: List[Dict[str, Any]],
        custom_metric_fn: Optional[callable] = None
    ) -> AggregatedResults:
        """Evaluate a batch of test cases.

        Args:
            test_data: List of test cases, each with:
                - question: str
                - answer: str
                - contexts: List[str]
                - ground_truth: str (optional)
                - metadata: dict (optional)
            custom_metric_fn: Optional function to compute custom metrics

        Returns:
            AggregatedResults with all metrics aggregated
        """
        results = []

        for idx, test_case in enumerate(test_data):
            logger.info("evaluating_test_case", index=idx, total=len(test_data))

            # Compute custom metrics if function provided
            custom_metrics = {}
            if custom_metric_fn:
                try:
                    custom_metrics = custom_metric_fn(test_case)
                except Exception as e:
                    logger.error("custom_metrics_failed", error=str(e), index=idx)

            # Evaluate with RAGAS
            result = self.evaluate_single(
                question=test_case['question'],
                answer=test_case['answer'],
                contexts=test_case.get('contexts', []),
                ground_truth=test_case.get('ground_truth'),
                custom_metrics=custom_metrics,
                metadata=test_case.get('metadata', {})
            )

            results.append(result)

        # Aggregate results
        aggregated = self._aggregate_results(results)

        return aggregated

    def _aggregate_results(self, results: List[EvaluationResult]) -> AggregatedResults:
        """Aggregate individual results into summary statistics.

        Args:
            results: List of evaluation results

        Returns:
            AggregatedResults with aggregated metrics
        """
        import numpy as np

        metrics_dict = {}

        # Aggregate RAGAS metrics
        ragas_metrics = [
            'context_relevance',
            'faithfulness',
            'answer_relevance',
            'answer_correctness'
        ]

        for metric_name in ragas_metrics:
            values = [
                getattr(r, metric_name)
                for r in results
                if getattr(r, metric_name) is not None
            ]

            if values:
                metrics_dict[metric_name] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }

        # Aggregate custom metrics
        custom_metric_names = set()
        for r in results:
            if r.custom_metrics:
                custom_metric_names.update(r.custom_metrics.keys())

        for metric_name in custom_metric_names:
            values = [
                r.custom_metrics[metric_name]
                for r in results
                if r.custom_metrics and metric_name in r.custom_metrics
            ]

            if values:
                metrics_dict[f"custom_{metric_name}"] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }

        return AggregatedResults(
            total_queries=len(results),
            metrics=metrics_dict,
            results=results,
            config=self.config
        )

    def export_results(
        self,
        results: AggregatedResults,
        output_path: str,
        format: str = 'json'
    ) -> None:
        """Export evaluation results to file.

        Args:
            results: Aggregated results to export
            output_path: Path to output file
            format: Export format ('json' or 'markdown')
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results.to_dict(), f, ensure_ascii=False, indent=2)

        elif format == 'markdown':
            self._export_markdown(results, output_file)

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info("results_exported", path=str(output_file), format=format)

    def _export_markdown(self, results: AggregatedResults, output_file: Path) -> None:
        """Export results as Markdown report.

        Args:
            results: Aggregated results
            output_file: Output file path
        """
        lines = [
            "# RAG System Evaluation Report",
            "",
            f"**Date:** {results.timestamp}",
            f"**Total Queries:** {results.total_queries}",
            "",
            "## Metrics Summary",
            ""
        ]

        # Add metrics table
        lines.append("| Metric | Mean | Median | Std | Min | Max | Count |")
        lines.append("|--------|------|--------|-----|-----|-----|-------|")

        for metric_name, stats in results.metrics.items():
            lines.append(
                f"| {metric_name} | "
                f"{stats['mean']:.3f} | "
                f"{stats['median']:.3f} | "
                f"{stats['std']:.3f} | "
                f"{stats['min']:.3f} | "
                f"{stats['max']:.3f} | "
                f"{stats['count']} |"
            )

        lines.append("")
        lines.append("## Individual Results")
        lines.append("")

        # Add individual results
        for idx, result in enumerate(results.results, 1):
            lines.append(f"### Query {idx}")
            lines.append(f"**Question:** {result.question}")
            lines.append(f"**Answer:** {result.answer[:200]}...")

            if result.ground_truth:
                lines.append(f"**Ground Truth:** {result.ground_truth[:200]}...")

            lines.append("")
            lines.append("**Metrics:**")

            if result.context_relevance is not None:
                lines.append(f"- Context Relevance: {result.context_relevance:.3f}")
            if result.faithfulness is not None:
                lines.append(f"- Faithfulness: {result.faithfulness:.3f}")
            if result.answer_relevance is not None:
                lines.append(f"- Answer Relevance: {result.answer_relevance:.3f}")
            if result.answer_correctness is not None:
                lines.append(f"- Answer Correctness: {result.answer_correctness:.3f}")

            for metric_name, score in result.custom_metrics.items():
                lines.append(f"- {metric_name}: {score:.3f}")

            lines.append("")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

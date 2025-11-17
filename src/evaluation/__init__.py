"""Evaluation module for RAG system.

This module provides comprehensive evaluation capabilities:
- RAGAS metrics integration (context relevance, faithfulness, answer relevance)
- Custom metrics for domain-specific evaluation
- Test dataset management
- Results aggregation and reporting
"""

from .evaluator import RAGASEvaluator, EvaluationResult
from .metrics import (
    CustomMetrics,
    ects_accuracy,
    reference_quality,
    abstaining_rate,
    hallucination_detection
)

__all__ = [
    'RAGASEvaluator',
    'EvaluationResult',
    'CustomMetrics',
    'ects_accuracy',
    'reference_quality',
    'abstaining_rate',
    'hallucination_detection'
]

#!/usr/bin/env python3
"""Custom evaluation metrics for domain-specific RAG evaluation."""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricResult:
    """Result of a custom metric evaluation.

    Attributes:
        score: Numeric score (0-1 typically)
        details: Additional details about the evaluation
        error: Error message if evaluation failed
    """
    score: float
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class CustomMetrics:
    """Container for all custom evaluation metrics."""

    @staticmethod
    def ects_accuracy(
        answer: str,
        ground_truth: str,
        tolerance: float = 0.0
    ) -> MetricResult:
        """Evaluate ECTS credit accuracy.

        Extracts ECTS values from answer and ground truth, compares them.

        Args:
            answer: Generated answer
            ground_truth: Expected answer
            tolerance: Acceptable difference in ECTS (default: exact match)

        Returns:
            MetricResult with score 1.0 if ECTS match (within tolerance), 0.0 otherwise

        Examples:
            >>> result = CustomMetrics.ects_accuracy("Das Modul hat 6 ECTS", "6 ECTS")
            >>> print(result.score)
            1.0
        """
        try:
            # Extract ECTS from answer
            answer_ects = _extract_ects(answer)

            # Extract ECTS from ground truth
            truth_ects = _extract_ects(ground_truth)

            if answer_ects is None and truth_ects is None:
                # No ECTS in either - not applicable
                return MetricResult(
                    score=1.0,
                    details={'reason': 'no_ects_mentioned', 'answer_ects': None, 'truth_ects': None}
                )

            if answer_ects is None or truth_ects is None:
                # ECTS in one but not the other - incorrect
                return MetricResult(
                    score=0.0,
                    details={'reason': 'missing_ects', 'answer_ects': answer_ects, 'truth_ects': truth_ects}
                )

            # Compare ECTS values
            difference = abs(answer_ects - truth_ects)

            if difference <= tolerance:
                return MetricResult(
                    score=1.0,
                    details={'answer_ects': answer_ects, 'truth_ects': truth_ects, 'difference': difference}
                )
            else:
                return MetricResult(
                    score=0.0,
                    details={'answer_ects': answer_ects, 'truth_ects': truth_ects, 'difference': difference}
                )

        except Exception as e:
            logger.error("ects_accuracy_failed", error=str(e))
            return MetricResult(score=0.0, error=str(e))

    @staticmethod
    def reference_quality(
        answer: str,
        contexts: List[str],
        required_citations: int = 1
    ) -> MetricResult:
        """Evaluate quality of source references in answer.

        Checks if answer contains proper citations and if cited information
        is actually present in the context.

        Args:
            answer: Generated answer
            contexts: Retrieved context chunks
            required_citations: Minimum number of citations required

        Returns:
            MetricResult with score based on citation quality

        Examples:
            >>> contexts = ["Das Modul hat 6 ECTS"]
            >>> answer = "Laut Quelle [1] hat das Modul 6 ECTS"
            >>> result = CustomMetrics.reference_quality(answer, contexts)
            >>> print(result.score >= 0.5)
            True
        """
        try:
            # Extract citation markers from answer
            citation_pattern = r'\[(\d+)\]'
            citations = re.findall(citation_pattern, answer)

            num_citations = len(set(citations))  # unique citations

            if num_citations == 0:
                return MetricResult(
                    score=0.0,
                    details={'num_citations': 0, 'required': required_citations, 'reason': 'no_citations'}
                )

            # Check if citations are valid (within range of contexts)
            max_valid_citation = len(contexts)
            invalid_citations = [
                int(c) for c in citations
                if int(c) > max_valid_citation or int(c) < 1
            ]

            if invalid_citations:
                score = 0.5  # Partial credit for having citations but some are invalid
                return MetricResult(
                    score=score,
                    details={
                        'num_citations': num_citations,
                        'invalid_citations': invalid_citations,
                        'reason': 'invalid_citation_numbers'
                    }
                )

            # Calculate score based on citation count vs requirement
            if num_citations >= required_citations:
                score = 1.0
            else:
                score = num_citations / required_citations

            return MetricResult(
                score=score,
                details={'num_citations': num_citations, 'required': required_citations}
            )

        except Exception as e:
            logger.error("reference_quality_failed", error=str(e))
            return MetricResult(score=0.0, error=str(e))

    @staticmethod
    def abstaining_rate(answers: List[str], abstaining_keywords: Optional[List[str]] = None) -> MetricResult:
        """Calculate rate of abstaining answers.

        Args:
            answers: List of generated answers
            abstaining_keywords: Keywords that indicate abstaining (default: German phrases)

        Returns:
            MetricResult with abstaining rate (0-1)

        Examples:
            >>> answers = ["Ich weiß nicht", "Das Modul hat 6 ECTS", "Kann ich nicht beantworten"]
            >>> result = CustomMetrics.abstaining_rate(answers)
            >>> print(result.score)  # Should be 2/3 ≈ 0.67
            0.666...
        """
        if abstaining_keywords is None:
            abstaining_keywords = [
                "kann nicht",
                "weiß nicht",
                "keine information",
                "nicht sicher",
                "kann ich nicht",
                "nicht beantworten",
                "keine antwort"
            ]

        try:
            abstaining_count = 0

            for answer in answers:
                answer_lower = answer.lower()

                # Check if any abstaining keyword is present
                if any(keyword in answer_lower for keyword in abstaining_keywords):
                    abstaining_count += 1

            rate = abstaining_count / len(answers) if answers else 0.0

            return MetricResult(
                score=rate,
                details={
                    'abstaining_count': abstaining_count,
                    'total_answers': len(answers),
                    'rate': rate
                }
            )

        except Exception as e:
            logger.error("abstaining_rate_failed", error=str(e))
            return MetricResult(score=0.0, error=str(e))

    @staticmethod
    def hallucination_detection(
        answer: str,
        contexts: List[str],
        check_keywords: Optional[List[str]] = None
    ) -> MetricResult:
        """Detect potential hallucinations in answer.

        Simple heuristic: Check if important facts in answer are grounded in context.

        Args:
            answer: Generated answer
            contexts: Retrieved context chunks
            check_keywords: Important keywords to check (default: ECTS, dates, numbers)

        Returns:
            MetricResult with hallucination score (0 = no hallucination, 1 = likely hallucination)

        Examples:
            >>> contexts = ["Das Modul hat 6 ECTS"]
            >>> answer = "Das Modul hat 12 ECTS"  # Hallucinated number
            >>> result = CustomMetrics.hallucination_detection(answer, contexts)
            >>> print(result.score > 0.5)  # Should detect hallucination
            True
        """
        if check_keywords is None:
            check_keywords = []

        try:
            # Combine all contexts
            combined_context = " ".join(contexts).lower()
            answer_lower = answer.lower()

            # Extract numbers from answer and context
            answer_numbers = set(re.findall(r'\d+', answer))
            context_numbers = set(re.findall(r'\d+', combined_context))

            # Check if answer contains numbers not in context
            hallucinated_numbers = answer_numbers - context_numbers

            # Extract ECTS values specifically
            answer_ects = _extract_ects(answer)
            context_ects_values = [_extract_ects(ctx) for ctx in contexts]
            context_ects_values = [e for e in context_ects_values if e is not None]

            ects_hallucinated = False
            if answer_ects is not None and answer_ects not in context_ects_values:
                ects_hallucinated = True

            # Calculate hallucination score
            hallucination_signals = 0

            if hallucinated_numbers:
                hallucination_signals += 1

            if ects_hallucinated:
                hallucination_signals += 1

            # Normalize score (0-1)
            max_signals = 2  # Currently checking 2 types of hallucinations
            score = hallucination_signals / max_signals

            return MetricResult(
                score=score,
                details={
                    'hallucinated_numbers': list(hallucinated_numbers),
                    'ects_hallucinated': ects_hallucinated,
                    'answer_ects': answer_ects,
                    'context_ects': context_ects_values
                }
            )

        except Exception as e:
            logger.error("hallucination_detection_failed", error=str(e))
            return MetricResult(score=0.0, error=str(e))


# Helper functions

def _extract_ects(text: str) -> Optional[float]:
    """Extract ECTS credit value from text.

    Args:
        text: Text to extract from

    Returns:
        ECTS value as float, or None if not found

    Examples:
        >>> _extract_ects("Das Modul hat 6 ECTS")
        6.0
        >>> _extract_ects("6 LP") # Leistungspunkte
        6.0
        >>> _extract_ects("Keine ECTS-Angabe")
        None
    """
    # Common patterns for ECTS
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:ECTS|ects|Ects)',
        r'(\d+(?:\.\d+)?)\s*(?:LP|lp|Lp)',  # Leistungspunkte
        r'(\d+(?:\.\d+)?)\s*(?:Credits|credits|Punkte)',
        r'(?:ECTS|LP|Credits):\s*(\d+(?:\.\d+)?)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue

    return None


# Standalone functions for backward compatibility and easier imports

def ects_accuracy(answer: str, ground_truth: str, tolerance: float = 0.0) -> float:
    """Standalone function for ECTS accuracy.

    Args:
        answer: Generated answer
        ground_truth: Expected answer
        tolerance: Acceptable difference

    Returns:
        Score (0-1)
    """
    result = CustomMetrics.ects_accuracy(answer, ground_truth, tolerance)
    return result.score


def reference_quality(answer: str, contexts: List[str], required_citations: int = 1) -> float:
    """Standalone function for reference quality.

    Args:
        answer: Generated answer
        contexts: Retrieved contexts
        required_citations: Required number of citations

    Returns:
        Score (0-1)
    """
    result = CustomMetrics.reference_quality(answer, contexts, required_citations)
    return result.score


def abstaining_rate(answers: List[str]) -> float:
    """Standalone function for abstaining rate.

    Args:
        answers: List of answers

    Returns:
        Abstaining rate (0-1)
    """
    result = CustomMetrics.abstaining_rate(answers)
    return result.score


def hallucination_detection(answer: str, contexts: List[str]) -> float:
    """Standalone function for hallucination detection.

    Args:
        answer: Generated answer
        contexts: Retrieved contexts

    Returns:
        Hallucination score (0-1, higher = more hallucination)
    """
    result = CustomMetrics.hallucination_detection(answer, contexts)
    return result.score

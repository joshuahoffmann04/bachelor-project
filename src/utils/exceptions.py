#!/usr/bin/env python3
"""Custom exceptions for RAG system.

This module defines custom exception types for better error handling
and more informative error messages throughout the system.
"""


class RAGSystemError(Exception):
    """Base exception for all RAG system errors."""
    pass


# Configuration Errors
class ConfigurationError(RAGSystemError):
    """Raised when configuration is invalid or missing."""
    pass


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when configuration file cannot be found."""
    pass


# LLM Errors
class LLMError(RAGSystemError):
    """Base exception for LLM-related errors."""
    pass


class LLMAPIError(LLMError):
    """Raised when LLM API call fails."""

    def __init__(self, provider: str, message: str, status_code: int = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"{provider} API error: {message}")


class LLMRateLimitError(LLMError):
    """Raised when LLM API rate limit is exceeded."""

    def __init__(self, provider: str, retry_after: int = None):
        self.provider = provider
        self.retry_after = retry_after
        msg = f"{provider} rate limit exceeded"
        if retry_after:
            msg += f" (retry after {retry_after}s)"
        super().__init__(msg)


class LLMTimeoutError(LLMError):
    """Raised when LLM API request times out."""

    def __init__(self, provider: str, timeout: int):
        self.provider = provider
        self.timeout = timeout
        super().__init__(f"{provider} request timed out after {timeout}s")


class LLMAuthenticationError(LLMError):
    """Raised when LLM API authentication fails."""

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"{provider} authentication failed - check API key")


# Retrieval Errors
class RetrievalError(RAGSystemError):
    """Base exception for retrieval-related errors."""
    pass


class IndexNotFoundError(RetrievalError):
    """Raised when retrieval index cannot be found."""

    def __init__(self, index_path: str):
        self.index_path = index_path
        super().__init__(f"Index not found: {index_path}")


class IndexCorruptedError(RetrievalError):
    """Raised when retrieval index is corrupted."""

    def __init__(self, index_path: str, reason: str = None):
        self.index_path = index_path
        msg = f"Index corrupted: {index_path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class EmbeddingError(RetrievalError):
    """Raised when embedding generation fails."""

    def __init__(self, text: str = None, reason: str = None):
        msg = "Embedding generation failed"
        if reason:
            msg += f": {reason}"
        if text:
            msg += f" (text length: {len(text)})"
        super().__init__(msg)


# Document Processing Errors
class DocumentProcessingError(RAGSystemError):
    """Base exception for document processing errors."""
    pass


class PDFParsingError(DocumentProcessingError):
    """Raised when PDF parsing fails."""

    def __init__(self, pdf_path: str, reason: str = None):
        self.pdf_path = pdf_path
        msg = f"Failed to parse PDF: {pdf_path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class ChunkingError(DocumentProcessingError):
    """Raised when text chunking fails."""

    def __init__(self, doc_id: str, reason: str = None):
        self.doc_id = doc_id
        msg = f"Failed to chunk document: {doc_id}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


# Web Scraping Errors
class ScrapingError(RAGSystemError):
    """Base exception for web scraping errors."""
    pass


class WebFetchError(ScrapingError):
    """Raised when fetching web content fails."""

    def __init__(self, url: str, status_code: int = None, reason: str = None):
        self.url = url
        self.status_code = status_code
        msg = f"Failed to fetch URL: {url}"
        if status_code:
            msg += f" (status: {status_code})"
        if reason:
            msg += f" - {reason}"
        super().__init__(msg)


class RobotsTxtError(ScrapingError):
    """Raised when robots.txt disallows scraping."""

    def __init__(self, url: str):
        self.url = url
        super().__init__(f"Scraping disallowed by robots.txt: {url}")


# Evaluation Errors
class EvaluationError(RAGSystemError):
    """Base exception for evaluation errors."""
    pass


class TestDatasetError(EvaluationError):
    """Raised when test dataset is invalid or cannot be loaded."""

    def __init__(self, dataset_path: str, reason: str = None):
        self.dataset_path = dataset_path
        msg = f"Invalid test dataset: {dataset_path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class MetricComputationError(EvaluationError):
    """Raised when metric computation fails."""

    def __init__(self, metric_name: str, reason: str = None):
        self.metric_name = metric_name
        msg = f"Failed to compute metric: {metric_name}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


# Storage Errors
class StorageError(RAGSystemError):
    """Base exception for storage-related errors."""
    pass


class CacheError(StorageError):
    """Raised when cache operations fail."""

    def __init__(self, cache_key: str, operation: str, reason: str = None):
        self.cache_key = cache_key
        self.operation = operation
        msg = f"Cache {operation} failed for key: {cache_key}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


# Validation Errors
class ValidationError(RAGSystemError):
    """Raised when input validation fails."""

    def __init__(self, field: str, value: any, reason: str):
        self.field = field
        self.value = value
        super().__init__(f"Validation failed for {field}: {reason}")


# Helper functions for exception handling
def is_recoverable_error(exc: Exception) -> bool:
    """Check if an error is recoverable (should retry).

    Args:
        exc: Exception to check

    Returns:
        True if error is recoverable, False otherwise
    """
    recoverable_types = (
        LLMRateLimitError,
        LLMTimeoutError,
        WebFetchError,
        CacheError
    )
    return isinstance(exc, recoverable_types)


def get_retry_delay(exc: Exception) -> int:
    """Get recommended retry delay for an exception.

    Args:
        exc: Exception to check

    Returns:
        Recommended delay in seconds, or 0 if not applicable
    """
    if isinstance(exc, LLMRateLimitError) and exc.retry_after:
        return exc.retry_after
    elif isinstance(exc, LLMTimeoutError):
        return 5
    elif isinstance(exc, WebFetchError):
        return 2
    return 0

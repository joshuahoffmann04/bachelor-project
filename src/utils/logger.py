"""DSGVO-compliant structured logging."""

import sys
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

import structlog
from structlog.processors import JSONRenderer
from structlog.stdlib import add_log_level, add_logger_name


def _hash_query(query: str) -> str:
    """Hash user query for privacy compliance.

    Args:
        query: User query string

    Returns:
        SHA256 hash of query
    """
    return hashlib.sha256(query.encode()).hexdigest()[:16]


def _anonymize_processor(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """Anonymize sensitive data in logs.

    Removes or hashes any potentially personal information.
    """
    # Hash user queries if present
    if 'query' in event_dict:
        event_dict['query_hash'] = _hash_query(event_dict['query'])
        del event_dict['query']

    # Remove IP addresses if accidentally logged
    if 'ip' in event_dict:
        del event_dict['ip']

    # Remove any email addresses
    if 'email' in event_dict:
        del event_dict['email']

    return event_dict


def _add_timestamp(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """Add ISO timestamp to log events."""
    event_dict['timestamp'] = datetime.utcnow().isoformat() + 'Z'
    return event_dict


def setup_logger(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    enable_privacy: bool = True
) -> None:
    """Setup structured logging with DSGVO compliance.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Output format ('json' or 'console')
        log_file: Optional file path for log output
        enable_privacy: Enable DSGVO-compliant anonymization
    """
    # Configure processors
    processors = [
        add_log_level,
        add_logger_name,
        _add_timestamp,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Add privacy processor if enabled
    if enable_privacy:
        processors.append(_anonymize_processor)

    # Add renderer based on format
    if log_format == 'json':
        processors.append(JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logging.root.addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class MetricsLogger:
    """Logger for performance and quality metrics.

    Provides methods for logging common metrics in a structured way.
    """

    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger

    def log_retrieval(
        self,
        query_hash: str,
        num_results: int,
        latency_ms: float,
        top_score: float,
        method: str = "hybrid"
    ) -> None:
        """Log retrieval metrics."""
        self.logger.info(
            "retrieval_completed",
            query_hash=query_hash,
            num_results=num_results,
            latency_ms=latency_ms,
            top_score=top_score,
            method=method
        )

    def log_generation(
        self,
        query_hash: str,
        latency_ms: float,
        tokens: int,
        provider: str,
        model: str
    ) -> None:
        """Log generation metrics."""
        self.logger.info(
            "generation_completed",
            query_hash=query_hash,
            latency_ms=latency_ms,
            tokens=tokens,
            provider=provider,
            model=model
        )

    def log_evaluation(
        self,
        query_hash: str,
        metrics: Dict[str, float]
    ) -> None:
        """Log evaluation metrics."""
        self.logger.info(
            "evaluation_completed",
            query_hash=query_hash,
            **metrics
        )

    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict] = None
    ) -> None:
        """Log error with context."""
        self.logger.error(
            "error_occurred",
            error_type=error_type,
            error_message=error_message,
            **(context or {})
        )

    def log_cache_hit(self, cache_type: str, key_hash: str) -> None:
        """Log cache hit."""
        self.logger.debug(
            "cache_hit",
            cache_type=cache_type,
            key_hash=key_hash
        )

    def log_cache_miss(self, cache_type: str, key_hash: str) -> None:
        """Log cache miss."""
        self.logger.debug(
            "cache_miss",
            cache_type=cache_type,
            key_hash=key_hash
        )

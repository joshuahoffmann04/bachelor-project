#!/usr/bin/env python3
"""Global constants for RAG system.

This module centralizes all magic numbers and constant values used throughout
the codebase, improving maintainability and making it easier to tune system behavior.
"""

# Chunking Constants
DEFAULT_CHUNK_SIZE = 512
"""Default chunk size in characters for text chunking."""

DEFAULT_CHUNK_OVERLAP = 50
"""Default overlap between chunks in characters."""

MIN_CHUNK_SIZE = 100
"""Minimum size for a text chunk in characters."""

MAX_CHUNK_SIZE = 1024
"""Maximum size for a text chunk in characters."""

TABLE_MAX_SIZE = 1024
"""Maximum size for table chunks in characters."""

AVG_WORD_LENGTH_CHARS = 6
"""Average word length in characters (used for word-to-char conversions).

This is used when converting between word-based and character-based measurements.
Derived from analysis of German text where average word length is ~5.5-6 characters.
"""

# Token Estimation Constants
TOKEN_APPROXIMATION_FACTOR = 1.3
"""Multiplier to approximate token count from word count.

When tiktoken is unavailable, we estimate: tokens ≈ words * 1.3
This is based on empirical observation that English/German text typically
has 1.3 tokens per word on average with GPT tokenizers.
"""

# Retrieval Constants
RRF_RANK_CONSTANT = 60
"""Constant k for Reciprocal Rank Fusion (RRF) algorithm.

RRF formula: score = sum(1 / (k + rank))
The value of 60 is a commonly used default that provides good balance.
Lower values give more weight to top-ranked items, higher values smooth the distribution.

Reference: Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet and
individual Rank Learning Methods" (SIGIR 2009)
"""

RETRIEVAL_OVERSAMPLING_FACTOR = 2
"""Factor by which to oversample during initial retrieval.

When retrieving top-k results, we initially retrieve k * OVERSAMPLING_FACTOR
results to allow for reranking, deduplication, or filtering.
"""

DEFAULT_TOP_K = 10
"""Default number of results to retrieve per query."""

DEFAULT_FINAL_TOP_K = 7
"""Default number of final results after fusion/reranking."""

# LLM Constants
DEFAULT_TEMPERATURE = 0.1
"""Default LLM temperature for generation (low for factual responses)."""

DEFAULT_MAX_TOKENS = 1000
"""Default maximum tokens for LLM generation."""

MAX_RETRIES = 3
"""Maximum number of retries for API calls."""

RETRY_DELAY_SECONDS = 2
"""Base delay in seconds between retries (may use exponential backoff)."""

# Evaluation Constants
ECTS_TOLERANCE = 0.0
"""Tolerance for ECTS value comparison (0 = exact match required)."""

REQUIRED_CITATIONS_MIN = 1
"""Minimum number of citations required in answers."""

# Web Scraping Constants
DEFAULT_REQUEST_TIMEOUT = 30
"""Default timeout for web requests in seconds."""

DEFAULT_RATE_LIMIT = 1
"""Default rate limit (requests per second) for web scraping."""

DEFAULT_CACHE_EXPIRY_SECONDS = 86400
"""Default cache expiry time in seconds (24 hours)."""

# File Size Limits
MAX_PDF_SIZE_MB = 50
"""Maximum PDF file size to process in megabytes."""

MAX_SCRAPED_CONTENT_SIZE_MB = 10
"""Maximum size of scraped web content in megabytes."""

# Logging and Monitoring
LOG_RETENTION_DAYS = 30
"""Number of days to retain log files."""

METRICS_AGGREGATION_WINDOW_SECONDS = 300
"""Time window for metrics aggregation in seconds (5 minutes)."""

# Performance Constants
DEFAULT_BATCH_SIZE = 32
"""Default batch size for embedding generation."""

MAX_WORKERS = 4
"""Maximum number of parallel workers for processing."""

MAX_MEMORY_GB = 8
"""Maximum memory usage limit in gigabytes."""

# Similarity Thresholds
MIN_SIMILARITY_THRESHOLD = 0.0
"""Minimum similarity score for retrieval (0.0 = no filtering)."""

HIGH_CONFIDENCE_THRESHOLD = 0.75
"""Threshold for high-confidence answers (abstaining threshold)."""

MEDIUM_CONFIDENCE_THRESHOLD = 0.5
"""Threshold for medium-confidence answers."""

LOW_CONFIDENCE_THRESHOLD = 0.25
"""Threshold for low-confidence answers."""

# Text Processing
MIN_PARAGRAPH_LENGTH = 10
"""Minimum length for a paragraph to be considered valid."""

MAX_QUERY_LENGTH = 500
"""Maximum length for user queries in characters."""

# Special Characters and Patterns
PARAGRAPH_SEPARATOR = '\n\n'
"""Separator used to split text into paragraphs."""

SECTION_SEPARATOR = '\n---\n'
"""Separator used to split text into sections."""

# Version Information
SYSTEM_VERSION = '1.0.0'
"""Current version of the RAG system."""

CONFIG_VERSION = '1.0.0'
"""Current version of the configuration format."""

# Feature Flags
ENABLE_EXPERIMENTAL_FEATURES = False
"""Enable experimental features (default: disabled for stability)."""

ENABLE_DETAILED_LOGGING = False
"""Enable detailed logging (may impact performance)."""

ENABLE_PROFILING = False
"""Enable performance profiling (development only)."""

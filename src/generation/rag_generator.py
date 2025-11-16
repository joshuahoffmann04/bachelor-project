"""RAG generator combining retrieval and generation."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time

from .llm_provider import LLMProvider, create_llm_provider
from ..retrieval.hybrid_retriever import HybridRetriever
from ..preprocessing.document import DocumentChunk
from ..utils.logger import get_logger, MetricsLogger

logger = get_logger(__name__)
metrics = MetricsLogger(logger)


class RAGGenerator:
    """RAG system combining retrieval and generation."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize RAG generator.

        Args:
            retriever: Hybrid retriever instance
            llm_provider: LLM provider (will be created from config if None)
            config: Configuration dictionary
        """
        self.config = config or {}
        self.retriever = retriever

        # Create LLM provider if not provided
        if llm_provider is None:
            self.llm_provider = create_llm_provider(self.config)
        else:
            self.llm_provider = llm_provider

        # Load prompts
        self.system_prompt = self._load_system_prompt()
        self.user_prompt_template = self._load_user_prompt_template()

        # Settings
        prompt_config = self.config.get('prompts', {})
        self.citation_required = prompt_config.get('citation', {}).get('required', True)
        self.abstaining_threshold = prompt_config.get('abstaining', {}).get('threshold', 0.6)
        self.abstaining_message = prompt_config.get('abstaining', {}).get(
            'message',
            'Ich kann diese Frage nicht sicher auf Basis der vorliegenden Dokumente beantworten.'
        )

    def _load_system_prompt(self) -> str:
        """Load system prompt from file.

        Returns:
            System prompt text
        """
        prompt_config = self.config.get('prompts', {})
        prompt_file = prompt_config.get('system_prompt_file', 'config/prompts/system_prompt.txt')

        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("system_prompt_not_found", file=prompt_file)
            return self._default_system_prompt()

    def _load_user_prompt_template(self) -> str:
        """Load user prompt template from file.

        Returns:
            User prompt template
        """
        prompt_config = self.config.get('prompts', {})
        template_file = prompt_config.get(
            'user_prompt_template_file',
            'config/prompts/user_prompt_template.txt'
        )

        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("user_prompt_template_not_found", file=template_file)
            return self._default_user_prompt_template()

    def _default_system_prompt(self) -> str:
        """Default system prompt if file not found."""
        return """Du bist ein präziser Assistent für Studienordnungen.

Regeln:
1. Antworte NUR basierend auf dem Kontext
2. IMMER Quellenangaben
3. Bei Unsicherheit: Verweigere die Antwort
4. Keine Halluzinationen"""

    def _default_user_prompt_template(self) -> str:
        """Default user prompt template if file not found."""
        return """KONTEXT: {context}

METADATEN: {source_metadata}

FRAGE: {question}

ANTWORT:"""

    async def generate(self, query: str) -> Dict[str, Any]:
        """Generate answer to query using RAG.

        Args:
            query: User query

        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()

        # Retrieve relevant chunks
        retrieval_start = time.time()
        chunks_with_scores = self.retriever.search(query)
        retrieval_time = (time.time() - retrieval_start) * 1000

        if not chunks_with_scores:
            return {
                'answer': self.abstaining_message,
                'sources': [],
                'confidence': 0.0,
                'metadata': {
                    'retrieval_time_ms': retrieval_time,
                    'generation_time_ms': 0,
                    'total_time_ms': (time.time() - start_time) * 1000
                }
            }

        # Check confidence
        avg_score = sum(score for _, score in chunks_with_scores) / len(chunks_with_scores)
        if avg_score < self.abstaining_threshold:
            logger.info("low_confidence_abstaining", avg_score=avg_score)
            return {
                'answer': self.abstaining_message,
                'sources': [chunk.get_citation() for chunk, _ in chunks_with_scores],
                'confidence': avg_score,
                'metadata': {
                    'retrieval_time_ms': retrieval_time,
                    'generation_time_ms': 0,
                    'total_time_ms': (time.time() - start_time) * 1000,
                    'abstained': True
                }
            }

        # Prepare context
        context = self._format_context(chunks_with_scores)
        source_metadata = self._format_source_metadata(chunks_with_scores)

        # Build prompt
        user_prompt = self.user_prompt_template.format(
            context=context,
            source_metadata=source_metadata,
            question=query
        )

        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        # Generate answer
        generation_start = time.time()
        try:
            answer = await self.llm_provider.generate(messages)
            generation_time = (time.time() - generation_start) * 1000

            # Log metrics
            query_hash = logger.bind().new().info.__self__._context.get('query_hash', 'unknown')
            metrics.log_generation(
                query_hash=query_hash,
                latency_ms=generation_time,
                tokens=self.llm_provider.count_tokens(answer),
                provider=self.config.get('llm', {}).get('provider', 'unknown'),
                model=getattr(self.llm_provider, 'model', 'unknown')
            )

            return {
                'answer': answer,
                'sources': [chunk.get_citation() for chunk, _ in chunks_with_scores],
                'chunks': [chunk for chunk, _ in chunks_with_scores],
                'scores': [score for _, score in chunks_with_scores],
                'confidence': avg_score,
                'metadata': {
                    'retrieval_time_ms': retrieval_time,
                    'generation_time_ms': generation_time,
                    'total_time_ms': (time.time() - start_time) * 1000,
                    'num_chunks': len(chunks_with_scores)
                }
            }

        except Exception as e:
            logger.error("generation_failed", error=str(e))
            metrics.log_error("generation_error", str(e), {'query_length': len(query)})

            return {
                'answer': f"Fehler bei der Generierung: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'metadata': {
                    'error': str(e),
                    'retrieval_time_ms': retrieval_time,
                    'total_time_ms': (time.time() - start_time) * 1000
                }
            }

    def _format_context(self, chunks_with_scores: List[Tuple[DocumentChunk, float]]) -> str:
        """Format chunks as context for LLM.

        Args:
            chunks_with_scores: Retrieved chunks with scores

        Returns:
            Formatted context string
        """
        context_parts = []

        for idx, (chunk, score) in enumerate(chunks_with_scores, 1):
            citation = chunk.get_citation()
            context_parts.append(
                f"[{idx}] {chunk.content}\n({citation}, Relevanz: {score:.2f})"
            )

        return "\n\n".join(context_parts)

    def _format_source_metadata(self, chunks_with_scores: List[Tuple[DocumentChunk, float]]) -> str:
        """Format source metadata for LLM.

        Args:
            chunks_with_scores: Retrieved chunks with scores

        Returns:
            Formatted metadata string
        """
        metadata_parts = []

        for idx, (chunk, score) in enumerate(chunks_with_scores, 1):
            metadata = [
                f"Quelle [{idx}]:",
                f"  - Dokument: {chunk.source_doc or 'Unbekannt'}",
                f"  - Typ: {chunk.source_type.value}",
                f"  - Inhaltstyp: {chunk.content_type.value}",
            ]

            if chunk.page:
                metadata.append(f"  - Seite: {chunk.page}")
            if chunk.section:
                metadata.append(f"  - Abschnitt: {chunk.section}")
            if chunk.source_url:
                metadata.append(f"  - URL: {chunk.source_url}")
            if chunk.scraped_at:
                metadata.append(f"  - Abgerufen: {chunk.scraped_at.strftime('%Y-%m-%d')}")

            metadata.append(f"  - Relevanz-Score: {score:.3f}")

            metadata_parts.append("\n".join(metadata))

        return "\n\n".join(metadata_parts)

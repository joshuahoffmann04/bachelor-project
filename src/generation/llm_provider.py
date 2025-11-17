"""LLM provider abstraction layer for OpenAI, Claude, and vLLM."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os

import openai
from anthropic import Anthropic
import httpx

from ..utils.logger import get_logger

logger = get_logger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM provider.

        Args:
            config: Provider configuration
        """
        self.config = config or {}

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenAI provider.

        Args:
            config: OpenAI configuration
        """
        super().__init__(config)

        # Configuration
        openai_config = self.config.get('llm', {}).get('openai', {})
        api_key = openai_config.get('api_key') or os.getenv('OPENAI_API_KEY')

        if not api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = openai.AsyncOpenAI(api_key=api_key)

        self.model = openai_config.get('model', 'gpt-4-turbo-preview')
        self.temperature = openai_config.get('temperature', 0.1)
        self.max_tokens = openai_config.get('max_tokens', 1000)

        logger.info("openai_provider_initialized", model=self.model)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate response using OpenAI API.

        Args:
            messages: Conversation messages
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional OpenAI parameters

        Returns:
            Generated text
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs
            )

            content = response.choices[0].message.content
            logger.info(
                "openai_generation_complete",
                tokens=response.usage.total_tokens,
                model=self.model
            )

            return content

        except Exception as e:
            logger.error("openai_generation_failed", error=str(e))
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken.

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback: approximate
            return len(text.split()) * 1.3


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Claude provider.

        Args:
            config: Claude configuration
        """
        super().__init__(config)

        # Configuration
        claude_config = self.config.get('llm', {}).get('claude', {})
        api_key = claude_config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')

        if not api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = Anthropic(api_key=api_key)

        self.model = claude_config.get('model', 'claude-3-opus-20240229')
        self.temperature = claude_config.get('temperature', 0.1)
        self.max_tokens = claude_config.get('max_tokens', 1000)

        logger.info("claude_provider_initialized", model=self.model)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate response using Claude API.

        Args:
            messages: Conversation messages
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional Claude parameters

        Returns:
            Generated text
        """
        try:
            # Extract system message if present
            system_message = None
            user_messages = []

            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    user_messages.append(msg)

            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                messages=user_messages,
                system=system_message,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs
            )

            content = response.content[0].text
            logger.info(
                "claude_generation_complete",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                model=self.model
            )

            return content

        except Exception as e:
            logger.error("claude_generation_failed", error=str(e))
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens (approximate for Claude).

        Args:
            text: Text to count

        Returns:
            Approximate number of tokens
        """
        # Claude uses similar tokenization to GPT
        # Approximate as words * 1.3
        return int(len(text.split()) * 1.3)


class VLLMProvider(LLMProvider):
    """vLLM (local model) provider."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize vLLM provider.

        Args:
            config: vLLM configuration
        """
        super().__init__(config)

        # Configuration
        vllm_config = self.config.get('llm', {}).get('vllm', {})
        self.base_url = vllm_config.get('base_url', 'http://localhost:8000')
        self.model = vllm_config.get('model', 'meta-llama/Llama-2-70b-chat-hf')
        self.temperature = vllm_config.get('temperature', 0.1)
        self.max_tokens = vllm_config.get('max_tokens', 1000)

        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

        logger.info("vllm_provider_initialized", model=self.model, base_url=self.base_url)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate response using vLLM API.

        Args:
            messages: Conversation messages
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional vLLM parameters

        Returns:
            Generated text
        """
        try:
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)

            # Call vLLM API
            response = await self.client.post(
                "/v1/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature or self.temperature,
                    "max_tokens": max_tokens or self.max_tokens,
                    **kwargs
                }
            )

            response.raise_for_status()
            result = response.json()

            content = result['choices'][0]['text']
            logger.info(
                "vllm_generation_complete",
                tokens=result.get('usage', {}).get('total_tokens', 0),
                model=self.model
            )

            return content

        except Exception as e:
            logger.error("vllm_generation_failed", error=str(e))
            raise

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt format.

        Args:
            messages: Conversation messages

        Returns:
            Formatted prompt
        """
        prompt_parts = []

        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def count_tokens(self, text: str) -> int:
        """Count tokens (approximate).

        Args:
            text: Text to count

        Returns:
            Approximate number of tokens
        """
        return int(len(text.split()) * 1.3)


def create_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """Factory function to create LLM provider.

    Args:
        config: Configuration dictionary

    Returns:
        LLM provider instance

    Raises:
        ValueError: If provider is unknown
    """
    provider_name = config.get('llm', {}).get('provider', 'openai')

    if provider_name == 'openai':
        return OpenAIProvider(config)
    elif provider_name == 'claude':
        return ClaudeProvider(config)
    elif provider_name == 'vllm':
        return VLLMProvider(config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")

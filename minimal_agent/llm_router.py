# llm_router.py
"""
Minimal LLM router supporting Anthropic and OpenAI.
"""
import os
from typing import Optional, Iterator
import anthropic
import openai


class LLMRouter:
    """Simple LLM router with support for Claude and GPT models."""

    def __init__(self):
        """Initialize the router with API keys from environment."""
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        # Initialize clients
        self.anthropic_client = None
        self.openai_client = None

        if self.anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)

        if self.openai_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_key)

        # Default model
        self.default_model = os.getenv("DEFAULT_MODEL", "claude-3-5-sonnet-20241022")

    def _is_anthropic_model(self, model: str) -> bool:
        """Check if model is an Anthropic model."""
        return model.startswith("claude")

    def _is_openai_model(self, model: str) -> bool:
        """Check if model is an OpenAI model."""
        return model.startswith("gpt")

    def generate_completion(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str | Iterator[str]:
        """
        Generate a completion using the specified model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to DEFAULT_MODEL env var)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response

        Returns:
            Complete response string, or iterator of response chunks if streaming
        """
        if model is None:
            model = self.default_model

        # Route to appropriate provider
        if self._is_anthropic_model(model):
            return self._call_anthropic(messages, model, max_tokens, temperature, stream)
        elif self._is_openai_model(model):
            return self._call_openai(messages, model, max_tokens, temperature, stream)
        else:
            raise ValueError(f"Unsupported model: {model}")

    def _call_anthropic(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        temperature: float,
        stream: bool
    ) -> str | Iterator[str]:
        """Call Anthropic API."""
        if not self.anthropic_client:
            raise ValueError("Anthropic API key not configured")

        # Separate system messages from conversation
        system_messages = [msg["content"] for msg in messages if msg["role"] == "system"]
        conversation_messages = [msg for msg in messages if msg["role"] != "system"]

        system_prompt = "\n\n".join(system_messages) if system_messages else None

        if stream:
            # Streaming response
            def stream_generator():
                with self.anthropic_client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=conversation_messages
                ) as stream:
                    for text in stream.text_stream:
                        yield text

            return stream_generator()
        else:
            # Non-streaming response
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=conversation_messages
            )
            return response.content[0].text

    def _call_openai(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        temperature: float,
        stream: bool
    ) -> str | Iterator[str]:
        """Call OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")

        if stream:
            # Streaming response
            def stream_generator():
                stream = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            return stream_generator()
        else:
            # Non-streaming response
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content

    def list_available_providers(self) -> list[str]:
        """List which providers are configured."""
        providers = []
        if self.anthropic_client:
            providers.append("Anthropic (Claude)")
        if self.openai_client:
            providers.append("OpenAI (GPT)")
        return providers

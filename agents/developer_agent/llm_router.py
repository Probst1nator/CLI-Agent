# llm_router.py
"""
Minimal LLM router supporting Anthropic (Claude), Google (Gemini), and Ollama.
"""
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

import anthropic
import google.generativeai as genai
import ollama


@dataclass(frozen=True)
class ModelSpec:
    """Structured model specification with provider and model key."""
    provider: str  # "anthropic", "google", "ollama"
    model_key: str  # The actual model identifier


# Model fallback chain - tried in sequence, also serves as registry
FALLBACK_MODELS: list[ModelSpec] = [
    ModelSpec("google", "gemini-flash-lite-latest"),
    ModelSpec("google", "gemini-flash-latest"),
    ModelSpec("ollama", "qwen3:1.7b"),
    ModelSpec("anthropic", "claude-4-5-haiku"),
]


class LLMRouter:
    """Simple LLM router with support for Claude, Gemini, and Ollama models."""

    def __init__(self):
        """Initialize the router with API keys from environment."""
        self.google_key = os.getenv("GOOGLE_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        # Initialize clients (order: Google -> Ollama -> Anthropic)
        self.google_configured = False
        self.ollama_available = False
        self.anthropic_client = None

        if self.google_key:
            genai.configure(api_key=self.google_key)
            self.google_configured = True

        # Check if Ollama is available (no API key needed)
        try:
            ollama.list()
            self.ollama_available = True
        except Exception:
            self.ollama_available = False

        if self.anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)

    def _get_model_spec(self, model: str) -> ModelSpec:
        """Look up a model string in the fallback models list.

        Args:
            model: Model string (e.g., "claude-4-5-haiku", "gemini-flash-latest")

        Returns:
            ModelSpec with provider and model_key

        Raises:
            ValueError: If model is not in the fallback models list
        """
        for model_spec in FALLBACK_MODELS:
            if model_spec.model_key == model:
                return model_spec

        available = ", ".join(m.model_key for m in FALLBACK_MODELS)
        raise ValueError(
            f"Unknown model: {model}. "
            f"Available models: {available}"
        )

    def generate_completion(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str | Iterator[str]:
        """
        Generate a completion using the specified model or fallback chain.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (looks up in KNOWN_MODELS), or None to use fallback chain
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response

        Returns:
            Complete response string, or iterator of response chunks if streaming
        """
        # Determine which model(s) to try
        if model is not None:
            # Use specified model
            models_to_try = [self._get_model_spec(model)]
        else:
            # Use fallback chain
            models_to_try = FALLBACK_MODELS

        # Try models in sequence until one succeeds
        last_error = None
        for model_spec in models_to_try:
            try:
                # Check if provider is available
                if not self._is_provider_available(model_spec.provider):
                    continue

                # Route to appropriate provider
                if model_spec.provider == "google":
                    return self._call_google(messages, model_spec.model_key, max_tokens, temperature, stream)
                elif model_spec.provider == "ollama":
                    return self._call_ollama(messages, model_spec.model_key, max_tokens, temperature, stream)
                elif model_spec.provider == "anthropic":
                    return self._call_anthropic(messages, model_spec.model_key, max_tokens, temperature, stream)
                else:
                    raise ValueError(f"Unsupported provider: {model_spec.provider}")
            except Exception as e:
                last_error = e
                continue

        # All models failed
        if last_error:
            raise RuntimeError(f"All models failed. Last error: {last_error}")
        else:
            raise RuntimeError("No models available (all providers unconfigured)")

    def _is_provider_available(self, provider: str) -> bool:
        """Check if a provider is configured and available."""
        if provider == "google":
            return self.google_configured
        elif provider == "ollama":
            return self.ollama_available
        elif provider == "anthropic":
            return self.anthropic_client is not None
        return False

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

    def _call_google(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        temperature: float,
        stream: bool
    ) -> str | Iterator[str]:
        """Call Google (Gemini) API."""
        if not self.google_configured:
            raise ValueError("Google API key not configured")

        # Convert messages to Gemini format
        # Gemini uses 'user' and 'model' roles (not 'assistant')
        gemini_messages = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                # Use first system message as system_instruction
                if system_instruction is None:
                    system_instruction = msg["content"]
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})
            else:
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})

        # Create model instance
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        model_instance = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            system_instruction=system_instruction
        )

        if stream:
            # Streaming response
            def stream_generator():
                response = model_instance.generate_content(
                    gemini_messages,
                    stream=True
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text

            return stream_generator()
        else:
            # Non-streaming response
            response = model_instance.generate_content(gemini_messages)

            # Check if response was blocked or has no valid parts
            if not response.candidates:
                raise ValueError("Google API returned no candidates (likely blocked by safety filters)")

            candidate = response.candidates[0]

            # Check finish reason
            # 1 = SAFETY, 2 = RECITATION, 3 = OTHER
            if candidate.finish_reason == 1:
                raise ValueError(
                    "Response blocked by Google's safety filters. "
                    "Try rephrasing your request or use a different model."
                )
            elif candidate.finish_reason == 2:
                raise ValueError("Response blocked due to recitation concerns")
            elif candidate.finish_reason not in (0, None):  # 0 or None = success
                raise ValueError(f"Response generation stopped (finish_reason={candidate.finish_reason})")

            # Safely access text
            if not candidate.content or not candidate.content.parts:
                raise ValueError("Google API returned empty response")

            return response.text

    def _call_ollama(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        temperature: float,
        stream: bool
    ) -> str | Iterator[str]:
        """Call Ollama (local models)."""
        if not self.ollama_available:
            raise ValueError("Ollama is not available. Make sure Ollama is installed and running.")

        # Ollama uses the same message format as OpenAI
        if stream:
            # Streaming response
            def stream_generator():
                response = ollama.chat(
                    model=model,
                    messages=messages,
                    stream=True,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                )
                for chunk in response:
                    if chunk.get("message", {}).get("content"):
                        yield chunk["message"]["content"]

            return stream_generator()
        else:
            # Non-streaming response
            response = ollama.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            )
            return response["message"]["content"]

    def list_available_providers(self) -> list[str]:
        """List which providers are configured (order: Google -> Ollama -> Anthropic)."""
        providers = []
        if self.google_configured:
            providers.append("Google (Gemini)")
        if self.ollama_available:
            providers.append("Ollama (Local)")
        if self.anthropic_client:
            providers.append("Anthropic (Claude)")
        return providers

# llm_router.py
"""
Minimal LLM router supporting Anthropic (Claude), Google (Gemini), and Ollama.
"""
import os
from typing import Optional, Iterator
import anthropic
import google.generativeai as genai
import ollama


class LLMRouter:
    """Simple LLM router with support for Claude, Gemini, and Ollama models."""

    def __init__(self):
        """Initialize the router with API keys from environment."""
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")

        # Initialize clients
        self.anthropic_client = None
        self.google_configured = False
        self.ollama_available = False

        if self.anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)

        if self.google_key:
            genai.configure(api_key=self.google_key)
            self.google_configured = True

        # Check if Ollama is available (no API key needed)
        try:
            ollama.list()
            self.ollama_available = True
        except Exception:
            self.ollama_available = False

        # Default model - Claude Haiku 4.5
        self.default_model = os.getenv("DEFAULT_MODEL", "claude-haiku-4-5")

    def _is_anthropic_model(self, model: str) -> bool:
        """Check if model is an Anthropic model."""
        return model.startswith("claude")

    def _is_google_model(self, model: str) -> bool:
        """Check if model is a Google (Gemini) model."""
        return model.startswith("gemini")

    def _is_ollama_model(self, model: str) -> bool:
        """Check if model is an Ollama model."""
        # Ollama supports various models - common ones include llama, mistral, codellama, etc.
        ollama_prefixes = ["llama", "mistral", "codellama", "phi", "mixtral", "vicuna", "orca"]
        return any(model.startswith(prefix) for prefix in ollama_prefixes)

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
        elif self._is_google_model(model):
            return self._call_google(messages, model, max_tokens, temperature, stream)
        elif self._is_ollama_model(model):
            return self._call_ollama(messages, model, max_tokens, temperature, stream)
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
        """List which providers are configured."""
        providers = []
        if self.anthropic_client:
            providers.append("Anthropic (Claude)")
        if self.google_configured:
            providers.append("Google (Gemini)")
        if self.ollama_available:
            providers.append("Ollama (Local)")
        return providers

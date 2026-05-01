"""
Unified LLM client for Lantern.

Backends:
    - "ollama"    (default) — talks to a local Ollama server at http://localhost:11434
    - "anthropic"           — uses ANTHROPIC_API_KEY for Claude

Pick a backend via the LANTERN_BACKEND env var, or pass backend= explicitly:

    llm = LLM()                          # local Ollama, qwen2.5-coder:7b
    llm = LLM(backend="anthropic")       # Claude Sonnet 4.6
    llm = LLM(model="qwen2.5-coder:14b") # override the model only

Every later week imports this. The public API is .stream() and .complete() —
treat them as stable.
"""

from __future__ import annotations

import os
from typing import Iterator, Literal, Optional

Backend = Literal["ollama", "anthropic"]

DEFAULT_MODELS: dict[Backend, str] = {
    "ollama": "qwen2.5-coder:7b",
    "anthropic": "claude-sonnet-4-6",
}


class LLM:
    """Thin streaming wrapper that hides backend differences."""

    def __init__(
        self,
        model: Optional[str] = None,
        backend: Optional[Backend] = None,
    ) -> None:
        chosen: Backend = backend or os.getenv("LANTERN_BACKEND", "ollama")  # type: ignore[assignment]
        if chosen not in ("ollama", "anthropic"):
            raise ValueError(f"Unknown backend: {chosen!r}. Use 'ollama' or 'anthropic'.")
        self.backend: Backend = chosen
        self.model: str = model or os.getenv("LANTERN_MODEL") or DEFAULT_MODELS[self.backend]
        self._client = self._make_client()

    def _make_client(self):
        if self.backend == "ollama":
            from ollama import Client
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            return Client(host=host)
        from anthropic import Anthropic
        return Anthropic()

    def stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> Iterator[str]:
        """Yield text chunks as they arrive from the model."""
        if self.backend == "ollama":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            for chunk in self._client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
                stream=True,
            ):
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
            return

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str:
        """Non-streaming convenience: collect the full reply as one string."""
        return "".join(
            self.stream(
                prompt,
                temperature=temperature,
                system=system,
                max_tokens=max_tokens,
            )
        )

    def __repr__(self) -> str:
        return f"LLM(backend={self.backend!r}, model={self.model!r})"

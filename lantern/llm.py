"""
Unified LLM client for Lantern.

Backends:
    - "ollama"    (default) — talks to a local Ollama server at http://localhost:11434
    - "anthropic"           — uses ANTHROPIC_API_KEY for Claude

Pick a backend via the LANTERN_BACKEND env var, or pass backend= explicitly:

    llm = LLM()                          # local Ollama, qwen2.5-coder:7b
    llm = LLM(backend="anthropic")       # Claude Sonnet 4.6
    llm = LLM(model="qwen2.5-coder:14b") # override the model only

Public surface (week 2):
    llm.stream(prompt, *, temperature, system, max_tokens) -> Iterator[str]
    llm.complete(prompt, ...) -> str
    llm.structured(prompt, schema, ...) -> BaseModel  # validated Pydantic instance

Every later week imports this. Treat the public API as stable.
"""

from __future__ import annotations

import json
import os
from typing import Iterator, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

Backend = Literal["ollama", "anthropic"]

DEFAULT_MODELS: dict[Backend, str] = {
    "ollama": "qwen2.5-coder:7b",
    "anthropic": "claude-sonnet-4-6",
}

T = TypeVar("T", bound=BaseModel)


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

    # ------------------------------------------------------------------ stream

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

    # -------------------------------------------------------------- structured

    def structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        temperature: float = 0.0,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        retries: int = 1,
    ) -> T:
        """
        Return a validated Pydantic instance via backend-native structured output.

        - Ollama: uses the `format=<json-schema>` constraint (model output is
          guaranteed to be valid JSON matching the schema).
        - Anthropic: uses tool-use as extraction (model is forced to call a
          single tool whose input matches the schema).

        Retries once on validation failure, feeding the error back to the model.
        """
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            error_hint = (
                f"\n\nYour previous reply failed validation: {last_error}\n"
                "Return a corrected JSON object that matches the schema exactly."
                if last_error
                else ""
            )
            try:
                if self.backend == "ollama":
                    raw = self._ollama_structured(
                        prompt + error_hint, schema, temperature, system, max_tokens
                    )
                    return schema.model_validate_json(raw)
                data = self._anthropic_structured(
                    prompt + error_hint, schema, temperature, system, max_tokens
                )
                return schema.model_validate(data)
            except (ValidationError, json.JSONDecodeError) as e:
                last_error = e
        assert last_error is not None
        raise last_error

    def _ollama_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        temperature: float,
        system: Optional[str],
        max_tokens: int,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat(
            model=self.model,
            messages=messages,
            format=schema.model_json_schema(),
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response["message"]["content"]

    def _anthropic_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        temperature: float,
        system: Optional[str],
        max_tokens: int,
    ) -> dict:
        tool = {
            "name": "submit",
            "description": "Submit the structured result.",
            "input_schema": schema.model_json_schema(),
        }
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": "submit"},
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        for block in response.content:
            if block.type == "tool_use":
                return block.input  # type: ignore[return-value]
        raise RuntimeError("Anthropic returned no tool_use block")

    def __repr__(self) -> str:
        return f"LLM(backend={self.backend!r}, model={self.model!r})"

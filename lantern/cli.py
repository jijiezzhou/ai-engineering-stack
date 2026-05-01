"""
Lantern CLI — entry point for the cumulative capstone.

Each week adds capabilities here. Week 1 ships a single command: stream a prompt
to the configured LLM (Ollama by default, Anthropic via env var).

Usage:
    uv run lantern "Explain Python decorators in 3 lines"
    uv run lantern "Write a haiku" -t 0.0
    uv run lantern "How do I list?" --system "You are a French chef."

Backends:
    LANTERN_BACKEND=ollama     (default; requires a running Ollama server)
    LANTERN_BACKEND=anthropic  (requires ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import time

import typer
from rich.console import Console

from lantern.llm import LLM

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Lantern — a local-first coding agent. Week 1: a streaming prompt CLI.",
)
console = Console()


@app.command()
def main(
    prompt: str = typer.Argument(..., help="Prompt to send to the model."),
    temperature: float = typer.Option(
        0.7, "-t", "--temperature", min=0.0, max=2.0,
        help="Sampling temperature. 0.0 = deterministic, 1.5 = chaotic.",
    ),
    backend: str = typer.Option(
        None, "--backend",
        help="Override LANTERN_BACKEND. 'ollama' or 'anthropic'.",
    ),
    model: str = typer.Option(
        None, "--model",
        help="Override the default model for the chosen backend.",
    ),
    system: str = typer.Option(
        None, "--system",
        help="Optional system prompt.",
    ),
):
    """Stream a single completion from the configured LLM backend."""
    llm = LLM(model=model, backend=backend)
    console.print(
        f"[dim]→ {llm.backend}:{llm.model}  T={temperature}"
        + (f"  system={system!r}" if system else "")
        + "[/dim]\n"
    )

    started = time.perf_counter()
    n_chars = 0
    n_chunks = 0
    for chunk in llm.stream(prompt, temperature=temperature, system=system):
        console.print(chunk, end="", soft_wrap=True, highlight=False, markup=False)
        n_chars += len(chunk)
        n_chunks += 1

    elapsed = time.perf_counter() - started
    rate = n_chars / elapsed if elapsed > 0 else 0
    console.print(
        f"\n\n[dim]({n_chars} chars / {n_chunks} chunks in {elapsed:.2f}s "
        f"≈ {rate:.0f} chars/s)[/dim]"
    )


if __name__ == "__main__":
    app()

"""
Lantern CLI — entry point for the cumulative capstone.

Subcommands grow each week:

    uv run lantern chat "Explain Python decorators in 3 lines"
    uv run lantern summarize lantern/llm.py
    uv run lantern ask "Where is the LLM client defined?" --repo .

Backends:
    LANTERN_BACKEND=ollama     (default; requires a running Ollama server)
    LANTERN_BACKEND=anthropic  (requires ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console

from lantern.agent import ask as ask_fn
from lantern.llm import LLM
from lantern.summarize import summarize_file

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Lantern — a local-first coding agent.",
)
console = Console()


@app.command("chat")
def chat(
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
    """Stream a single completion from the configured LLM backend (week 1)."""
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


@app.command("summarize")
def summarize(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True,
                                 help="Path to the source file to summarize."),
    backend: str = typer.Option(None, "--backend",
                                 help="Override LANTERN_BACKEND."),
    model: str = typer.Option(None, "--model",
                               help="Override the default model."),
    json_out: bool = typer.Option(False, "--json",
                                   help="Emit raw JSON instead of pretty output."),
):
    """Produce a typed structured summary of a source file (week 2)."""
    llm = LLM(model=model, backend=backend)
    if not json_out:
        console.print(
            f"[dim]→ {llm.backend}:{llm.model}  summarizing {path}[/dim]"
        )

    started = time.perf_counter()
    summary = summarize_file(path, llm=llm)
    elapsed = time.perf_counter() - started

    if json_out:
        print(summary.model_dump_json())
        return

    console.print()
    console.print(f"[bold cyan]{summary.path}[/bold cyan]  [dim]({summary.language})[/dim]")
    console.print(f"[italic]{summary.one_liner}[/italic]")

    if summary.public_api:
        console.print("\n[bold]Public API[/bold]")
        for s in summary.public_api:
            console.print(f"  • {s}")
    if summary.dependencies:
        console.print("\n[bold]Dependencies[/bold]")
        for d in summary.dependencies:
            console.print(f"  • {d}")
    if summary.notable:
        console.print("\n[bold]Notable[/bold]")
        for n in summary.notable:
            console.print(f"  • {n}")

    console.print(
        f"\n[dim]confidence={summary.confidence:.2f}  ({elapsed:.2f}s)[/dim]"
    )


@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Question about the codebase."),
    repo: Path = typer.Option(
        Path("."), "-r", "--repo",
        exists=True, file_okay=False, dir_okay=True, readable=True,
        help="Repository root that tools may read from.",
    ),
    backend: str = typer.Option(None, "--backend",
                                 help="Override LANTERN_BACKEND."),
    model: str = typer.Option(None, "--model",
                               help="Override the default model."),
):
    """Ask a question; Lantern picks one tool, reads the code, then answers (week 3)."""
    llm = LLM(model=model, backend=backend)
    repo_resolved = repo.resolve()
    console.print(
        f"[dim]→ {llm.backend}:{llm.model}  repo={repo_resolved}[/dim]\n"
    )

    started = time.perf_counter()
    answer = ask_fn(question, repo=repo_resolved, llm=llm)
    elapsed = time.perf_counter() - started

    console.print(answer)
    console.print(f"\n[dim]({elapsed:.2f}s)[/dim]")


if __name__ == "__main__":
    app()

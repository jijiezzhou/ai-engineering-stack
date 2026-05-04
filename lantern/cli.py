"""
Lantern CLI — entry point for the cumulative capstone.

Subcommands grow each week:

    lantern chat "Explain Python decorators in 3 lines"             (week 1)
    lantern summarize lantern/llm.py                                (week 2)
    lantern ask "Where is the LLM client defined?" --repo .         (week 3)
    lantern index .                                                 (week 4)
    lantern search "where is path traversal blocked" --repo .       (week 4)

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


@app.command("index")
def index(
    repo: Path = typer.Argument(
        Path("."), exists=True, file_okay=False, dir_okay=True, readable=True,
        help="Repository root to index.",
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild",
        help="Wipe the existing collection and re-embed from scratch.",
    ),
):
    """Embed a repository's code chunks for semantic search (week 4)."""
    from lantern.index import index_repo, INDEX_ROOT

    repo_resolved = repo.resolve()
    console.print(
        f"[dim]→ indexing {repo_resolved}\n"
        f"  store: {INDEX_ROOT / repo_resolved.name}{'  (rebuild)' if rebuild else ''}[/dim]"
    )

    last_pct = -1

    def progress(done: int, total: int) -> None:
        nonlocal last_pct
        pct = int(done * 100 / total) if total else 100
        if pct >= last_pct + 10 or done == total:
            console.print(f"  embedded {done}/{total} chunks ({pct}%)")
            last_pct = pct

    started = time.perf_counter()
    stats = index_repo(repo_resolved, rebuild=rebuild, progress=progress)
    elapsed = time.perf_counter() - started

    console.print(
        f"\n[bold]Indexed {stats['files']} files / {stats['chunks']} chunks[/bold] "
        f"[dim]in {elapsed:.1f}s[/dim]"
    )


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Natural-language query."),
    repo: Path = typer.Option(
        Path("."), "-r", "--repo",
        exists=True, file_okay=False, dir_okay=True, readable=True,
        help="Repository whose index to search. Run `lantern index` first.",
    ),
    top_k: int = typer.Option(5, "-k", "--top-k", min=1, max=50,
                               help="How many hits to return."),
    retriever: str = typer.Option(
        "vector", "--retriever",
        help="Retriever to use: vector | bm25 | hybrid (week 5).",
    ),
):
    """Search an indexed repository (week 4 + week 5 retrievers)."""
    from lantern.search import search as vector_search, bm25_search, hybrid_search

    fns = {
        "vector": lambda q: vector_search(q, repo=repo.resolve(), top_k=top_k),
        "bm25":   lambda q: bm25_search(q, repo=repo.resolve(), top_k=top_k),
        "hybrid": lambda q: hybrid_search(q, repo=repo.resolve(), top_k=top_k),
    }
    if retriever not in fns:
        console.print(f"[red]Unknown retriever {retriever!r}; choices: {sorted(fns)}[/red]")
        raise typer.Exit(2)

    console.print(
        f"[dim]→ searching {repo.resolve()}  retriever={retriever}  query={query!r}[/dim]"
    )

    started = time.perf_counter()
    hits = fns[retriever](query)
    elapsed = time.perf_counter() - started

    if not hits:
        console.print("\n(no hits — did you run `lantern index .` first?)")
        return

    for h in hits:
        symbol = f" [yellow]{h.kind}[/yellow]:[bold]{h.name}[/bold]" if h.name else f" [yellow]{h.kind}[/yellow]"
        console.print(
            f"\n[cyan]{h.path}:{h.start_line}-{h.end_line}[/cyan]"
            f"{symbol}  [dim]score={h.score:.3f}[/dim]"
        )
        preview = "\n".join(h.content.splitlines()[:5])
        console.print(f"[dim]{preview}[/dim]")

    console.print(f"\n[dim]({len(hits)} hits in {elapsed:.2f}s)[/dim]")


@app.command("eval")
def eval_cmd(
    repo: Path = typer.Option(
        Path("."), "-r", "--repo",
        exists=True, file_okay=False, dir_okay=True, readable=True,
        help="Repository to evaluate retrievers against.",
    ),
    tests: Path = typer.Option(
        Path("evals/lantern.yaml"), "-t", "--tests",
        exists=True, dir_okay=False, readable=True,
        help="Path to the YAML test set.",
    ),
    top_k: int = typer.Option(5, "-k", "--top-k", min=1, max=50),
    skip_rerank: bool = typer.Option(
        False, "--skip-rerank",
        help="Skip the LLM-based reranker (saves ~10 s/question on local).",
    ),
):
    """Evaluate retrievers on a golden Q&A test set (week 5)."""
    from rich.table import Table
    from lantern.evals import evaluate, load_tests
    from lantern.search import bm25_search, hybrid_search, search as vector_search
    from lantern.rerank import rerank

    cases = load_tests(tests)
    repo_resolved = repo.resolve()
    console.print(
        f"[dim]→ evaluating {len(cases)} questions  repo={repo_resolved}  k={top_k}[/dim]\n"
    )

    fns: dict[str, callable] = {
        "vector": lambda q: vector_search(q, repo=repo_resolved, top_k=top_k),
        "bm25":   lambda q: bm25_search(q, repo=repo_resolved, top_k=top_k),
        "hybrid": lambda q: hybrid_search(q, repo=repo_resolved, top_k=top_k),
    }
    if not skip_rerank:
        llm = LLM()
        # Pool 4× the requested top_k so the reranker has enough candidates
        # to actually re-order. If the right answer isn't in the pool,
        # rerank can't surface it.
        fns["hybrid+rerank"] = lambda q: rerank(
            q,
            hybrid_search(q, repo=repo_resolved, top_k=top_k * 4),
            llm=llm,
            top_k=top_k,
        )

    reports = []
    for name, fn in fns.items():
        started = time.perf_counter()
        report = evaluate(name, fn, cases)
        elapsed = time.perf_counter() - started
        reports.append((report, elapsed))
        console.print(
            f"  [green]✓[/green] {name:<16} {elapsed:.1f}s "
            f"(R@1={report.recall_at(1):.2f} R@{top_k}={report.recall_at(top_k):.2f} MRR={report.mrr:.2f})"
        )

    table = Table(title=f"Eval — {len(cases)} questions, top-{top_k}")
    table.add_column("Retriever", style="cyan", no_wrap=True)
    table.add_column("Recall@1", justify="right")
    table.add_column("Recall@3", justify="right")
    table.add_column(f"Recall@{top_k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Time", justify="right", style="dim")
    for r, elapsed in reports:
        table.add_row(
            r.name,
            f"{r.recall_at(1):.2f}",
            f"{r.recall_at(3):.2f}",
            f"{r.recall_at(top_k):.2f}",
            f"{r.mrr:.2f}",
            f"{elapsed:.1f}s",
        )
    console.print()
    console.print(table)


if __name__ == "__main__":
    app()

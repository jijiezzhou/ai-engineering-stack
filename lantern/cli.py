"""
Lantern CLI — entry point for the cumulative capstone.

Subcommands grow each week:

    lantern chat "Explain Python decorators in 3 lines"             (week 1)
    lantern summarize lantern/llm.py                                (week 2)
    lantern ask "Where is the LLM client defined?" --repo .         (week 3 / 6)
    lantern index .                                                 (week 4)
    lantern search "where is path traversal blocked" --repo .       (week 4 / 5)
    lantern eval --skip-rerank                                      (week 5)
    lantern mcp                                                     (week 7)
    lantern trace                                                   (week 7)

Backends:
    LANTERN_BACKEND=ollama     (default; requires a running Ollama server)
    LANTERN_BACKEND=anthropic  (requires ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console

from lantern.agent import agent_loop, ask as ask_fn
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
    single_step: bool = typer.Option(
        False, "--single-step",
        help="Use the week-3 two-shot agent (one tool, then answer).",
    ),
    max_steps: int = typer.Option(
        8, "--max-steps", min=1, max=20,
        help="Maximum tool-using iterations before forcing a final answer.",
    ),
    no_retrieval: bool = typer.Option(
        False, "--no-retrieval",
        help="Skip the hybrid-search primer at step 0.",
    ),
    show_trace: bool = typer.Option(
        False, "--show-trace",
        help="Print the agent's reasoning + tool calls before the answer.",
    ),
    save_trace: bool = typer.Option(
        False, "--save-trace",
        help="Write a JSONL trace to ~/.lantern/traces/ (week 7).",
    ),
    backend: str = typer.Option(None, "--backend",
                                 help="Override LANTERN_BACKEND."),
    model: str = typer.Option(None, "--model",
                               help="Override the default model."),
):
    """Ask a question; the agent inspects the code and answers (week 6 multi-step by default)."""
    from lantern.trace import Trace

    llm = LLM(model=model, backend=backend)
    repo_resolved = repo.resolve()
    mode = "single-step" if single_step else f"multi-step (max={max_steps})"
    primer_note = "" if no_retrieval or single_step else "  retrieval=on"
    console.print(
        f"[dim]→ {llm.backend}:{llm.model}  repo={repo_resolved}  mode={mode}{primer_note}[/dim]\n"
    )

    started = time.perf_counter()
    trace_obj = Trace() if (save_trace and not single_step) else None
    if single_step:
        answer = ask_fn(question, repo=repo_resolved, llm=llm)
        steps = None
        forced = False
        run_id = None
    else:
        result = agent_loop(
            question,
            repo=repo_resolved,
            llm=llm,
            max_steps=max_steps,
            use_retrieval=not no_retrieval,
            trace=trace_obj,
        )
        answer = result.answer
        steps = result.steps
        forced = result.forced_final
        run_id = result.run_id
    elapsed = time.perf_counter() - started

    if show_trace and steps:
        console.print("[bold]Trace[/bold]")
        for i, step in enumerate(steps, 1):
            d = step.decision
            console.print(
                f"\n[cyan]Step {i}[/cyan]  [yellow]{d.next_action}[/yellow]"
                + (f"  path={d.path!r}" if d.path else "")
                + (f"  pattern={d.pattern!r}" if d.pattern else "")
            )
            if d.reasoning:
                console.print(f"  [dim]reasoning: {d.reasoning}[/dim]")
            if step.tool_output is not None:
                preview = step.tool_output.strip().splitlines()[:3]
                if preview:
                    console.print(f"  [dim]output: {' | '.join(s[:80] for s in preview)}[/dim]")
        console.print("\n[bold]Answer[/bold]")

    console.print(answer)
    suffix = "  (max steps reached)" if forced else ""
    n_steps = len(steps) if steps else 1
    console.print(f"\n[dim]({elapsed:.2f}s; {n_steps} step{'s' if n_steps != 1 else ''}{suffix})[/dim]")
    if run_id:
        console.print(f"[dim]trace saved: {run_id}  (lantern trace {run_id})[/dim]")


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
    kinds: str = typer.Option(
        None, "--kinds",
        help="Comma-separated chunk classes to include: code, doc, config, other (week 9). Default: all.",
    ),
):
    """Search an indexed repository (week 4 + week 5 retrievers; week 9 --kinds filter)."""
    from lantern.search import search as vector_search, bm25_search, hybrid_search

    kinds_list = [k.strip() for k in kinds.split(",") if k.strip()] if kinds else None

    fns = {
        "vector": lambda q: vector_search(q, repo=repo.resolve(), top_k=top_k, kinds=kinds_list),
        "bm25":   lambda q: bm25_search(q, repo=repo.resolve(), top_k=top_k, kinds=kinds_list),
        "hybrid": lambda q: hybrid_search(q, repo=repo.resolve(), top_k=top_k, kinds=kinds_list),
    }
    if retriever not in fns:
        console.print(f"[red]Unknown retriever {retriever!r}; choices: {sorted(fns)}[/red]")
        raise typer.Exit(2)

    kinds_note = f"  kinds={kinds_list}" if kinds_list else ""
    console.print(
        f"[dim]→ searching {repo.resolve()}  retriever={retriever}{kinds_note}  query={query!r}[/dim]"
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
        help="Repository to evaluate against.",
    ),
    tests: Path = typer.Option(
        Path("evals/lantern.yaml"), "-t", "--tests",
        exists=True, dir_okay=False, readable=True,
        help="Path to the YAML test set.",
    ),
    mode: str = typer.Option(
        "retrieval", "--mode",
        help="retrieval (week 5) or agent (week 8).",
    ),
    top_k: int = typer.Option(5, "-k", "--top-k", min=1, max=50),
    skip_rerank: bool = typer.Option(
        False, "--skip-rerank",
        help="Retrieval mode: skip the LLM reranker (~10s/question saved).",
    ),
    max_steps: int = typer.Option(
        8, "--max-steps", min=1, max=20,
        help="Agent mode: max steps per question.",
    ),
    questions: int = typer.Option(
        0, "--questions", min=0,
        help="Limit to first N questions (0 = all). Useful for slow agent mode.",
    ),
    judge_backend: str = typer.Option(
        None, "--judge-backend",
        help="Agent mode: backend for the LLM-as-judge ('ollama'|'anthropic'). Defaults to LANTERN_BACKEND.",
    ),
):
    """Evaluate retrievers (week 5) or the agent end-to-end (week 8)."""
    from rich.table import Table
    from lantern.evals import evaluate, evaluate_agent, load_tests

    cases = load_tests(tests)
    if questions:
        cases = cases[:questions]
    repo_resolved = repo.resolve()

    if mode == "retrieval":
        from lantern.search import bm25_search, hybrid_search, search as vector_search
        from lantern.rerank import rerank

        console.print(
            f"[dim]→ retrieval eval: {len(cases)} questions  repo={repo_resolved}  k={top_k}[/dim]\n"
        )

        fns: dict[str, callable] = {
            "vector": lambda q: vector_search(q, repo=repo_resolved, top_k=top_k),
            "bm25":   lambda q: bm25_search(q, repo=repo_resolved, top_k=top_k),
            "hybrid": lambda q: hybrid_search(q, repo=repo_resolved, top_k=top_k),
        }
        if not skip_rerank:
            llm = LLM()
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

        table = Table(title=f"Retrieval eval — {len(cases)} questions, top-{top_k}")
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
        return

    if mode == "agent":
        # Filter to cases with golden_answer
        cases_with_gold = [c for c in cases if c.golden_answer]
        if not cases_with_gold:
            console.print("[red]No test cases have a `golden_answer` field set.[/red]")
            raise typer.Exit(2)

        llm = LLM()
        judge = LLM(backend=judge_backend) if judge_backend else llm
        console.print(
            f"[dim]→ agent eval: {len(cases_with_gold)} questions on "
            f"{llm.backend}:{llm.model}  judge={judge.backend}:{judge.model}  "
            f"max_steps={max_steps}  repo={repo_resolved}[/dim]\n"
        )
        console.print("[dim](slow — multi-step Qwen 7B is ~30-90 s per question; ~10-30 min total)[/dim]\n")

        def on_case(case):
            mark = "[green]✓[/green]" if case.correct else "[red]✗[/red]"
            steps = f"{case.n_steps}st"
            if case.forced_final:
                steps += "*"
            q = case.question[:55]
            console.print(
                f"  {mark} {case.elapsed_s:>5.0f}s  {steps:<5} "
                f"conf={case.confidence:.2f}  {q}"
            )

        report = evaluate_agent(
            cases_with_gold,
            repo=repo_resolved,
            llm=llm,
            judge=judge,
            max_steps=max_steps,
            on_case=on_case,
        )

        table = Table(title=f"Agent eval — {len(report.cases)} questions  ({report.name})")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right")
        table.add_row("Correctness (LLM-judge)", f"{report.correctness:.2f}")
        table.add_row("Avg judge confidence", f"{report.avg_confidence:.2f}")
        table.add_row("File-hit rate (any expected file opened)", f"{report.file_hit_rate:.2f}")
        table.add_row("Avg steps per question", f"{report.avg_steps:.1f}")
        table.add_row("Forced-final rate", f"{report.forced_rate:.2f}")
        table.add_row("Total elapsed", f"{report.total_elapsed_s:.1f}s")
        console.print()
        console.print(table)
        return

    console.print(f"[red]Unknown --mode {mode!r}; choices: retrieval, agent[/red]")
    raise typer.Exit(2)


@app.command("mcp")
def mcp_cmd():
    """Run Lantern as an MCP server (stdio transport).

    Set LANTERN_REPO to the absolute path of the repo to expose. Wire into
    Claude Code's claude_desktop_config.json — see weeks/07-production/README.md.
    """
    from lantern.mcp import run as run_mcp
    run_mcp()


@app.command("trace")
def trace_cmd(
    run_id: str = typer.Argument(
        None,
        help="Run id (or trace filename) to display. If omitted, lists recent runs.",
    ),
    limit: int = typer.Option(20, "--limit", min=1, max=200,
                               help="How many recent runs to show in list mode."),
):
    """List or replay agent traces (week 7)."""
    from lantern.trace import TRACE_ROOT, iter_runs, read_run

    if run_id is None:
        runs = list(iter_runs(TRACE_ROOT))[-limit:]
        if not runs:
            console.print(f"[dim]no traces yet under {TRACE_ROOT}[/dim]")
            return
        from rich.table import Table
        table = Table(title=f"Recent traces (newest last) — {TRACE_ROOT}")
        table.add_column("run_id", style="cyan")
        table.add_column("steps", justify="right")
        table.add_column("final", style="dim")
        table.add_column("question")
        for _, summary in runs:
            q = (summary.get("question") or "")[:80]
            table.add_row(
                summary["run_id"],
                str(summary["num_steps"]),
                summary.get("final_kind") or "?",
                q,
            )
        console.print(table)
        return

    try:
        events = read_run(run_id)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    if not events:
        console.print("(empty trace)")
        return

    for ev in events:
        kind = ev.get("kind", "")
        ts = ev.get("ts", 0.0)
        step = ev.get("step", "-")
        if kind == "run_start":
            console.print(
                f"[bold]{ev.get('question')}[/bold]\n"
                f"[dim]repo={ev.get('repo')}  {ev.get('backend')}:{ev.get('model')}  "
                f"max_steps={ev.get('max_steps')}[/dim]"
            )
        elif kind == "primer":
            console.print(f"[dim]· primer: {ev.get('chars')} chars[/dim]")
        elif kind == "decision":
            console.print(
                f"\n[cyan]Step {step}[/cyan] [yellow]{ev.get('next_action')}[/yellow]"
                + (f"  path={ev.get('path')!r}" if ev.get("path") else "")
                + (f"  pattern={ev.get('pattern')!r}" if ev.get("pattern") else "")
                + f"  [dim]+{ts:.2f}s[/dim]"
            )
            console.print(f"  [dim]reasoning: {ev.get('reasoning')}[/dim]")
        elif kind == "tool_output":
            console.print(f"  [dim]output {ev.get('chars')} chars: {ev.get('preview', '')[:160]}[/dim]")
        elif kind == "answer":
            console.print(f"\n[bold]Answer[/bold]\n{ev.get('text', '')}")
            console.print(f"[dim]+{ts:.2f}s[/dim]")
        elif kind == "forced_final":
            console.print(f"\n[bold]Answer (forced — max steps reached)[/bold]\n{ev.get('text', '')}")
            console.print(f"[dim]+{ts:.2f}s[/dim]")


if __name__ == "__main__":
    app()

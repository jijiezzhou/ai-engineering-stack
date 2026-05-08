"""
Week 7 — Lantern as an MCP server.

Exposes Lantern's capabilities as Model Context Protocol tools so any
MCP-aware client (Claude Code, Cursor, Goose, ...) can use them. Stdio
transport — the simplest to wire into a desktop client.

Operates on a single repository per server instance, configured via the
`LANTERN_REPO` env var (defaults to the cwd at server start). Run multiple
servers if you need to expose multiple repos.

Run with: `lantern mcp`

Wire into Claude Code by adding to `~/Library/Application Support/Claude/
claude_desktop_config.json`:

    {
      "mcpServers": {
        "lantern": {
          "command": "lantern",
          "args": ["mcp"],
          "env": {"LANTERN_REPO": "/absolute/path/to/some/repo"}
        }
      }
    }

Then restart Claude Code. Lantern's tools appear in the tool list.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Resolve the repo at import time so each spawned server is pinned to one
# project. Path-traversal protection inside read_file/list_dir/grep is what
# actually enforces the boundary; this just sets the root.
_REPO = Path(os.getenv("LANTERN_REPO", os.getcwd())).resolve()


server = FastMCP("lantern")


@server.tool()
def about_repo() -> str:
    """Tell the client which repository this Lantern server is exposing."""
    return f"Lantern is serving the repository at: {_REPO}"


@server.tool()
def read_file(path: str) -> str:
    """Read a file inside the configured repository.

    `path` is relative to the repo root. Returns the file's text contents
    (truncated to 16KB). Refuses paths that resolve outside the repo."""
    from lantern.tools import read_file as rf
    return rf(path=path).run(_REPO)


@server.tool()
def list_dir(path: str = ".") -> str:
    """List immediate children of a directory inside the configured repo.

    Hides build artefacts (.git, __pycache__, node_modules, etc.)."""
    from lantern.tools import list_dir as ld
    return ld(path=path).run(_REPO)


@server.tool()
def grep(pattern: str, path: str = ".") -> str:
    """Find lines containing `pattern` (case-insensitive substring) in
    files under `path`. Returns rows of `file:line: content`."""
    from lantern.tools import grep as gp
    return gp(pattern=pattern, path=path).run(_REPO)


@server.tool()
def search(query: str, top_k: int = 5) -> str:
    """Hybrid semantic + BM25 search over the configured repo's index.

    Run `lantern index <repo>` first to build the index. Returns the top
    hits as `path:line-range (kind:name) score: <preview>`."""
    from lantern.search import hybrid_search

    try:
        hits = hybrid_search(query, repo=_REPO, top_k=top_k)
    except FileNotFoundError as e:
        return f"ERROR: {e}"
    if not hits:
        return "(no hits)"

    out: list[str] = []
    for h in hits:
        try:
            rel = Path(h.path).resolve().relative_to(_REPO)
        except (ValueError, OSError):
            rel = Path(h.path).name
        kind = f"{h.kind}:{h.name}" if h.name else h.kind
        preview = " | ".join(s.strip()[:80] for s in h.content.splitlines()[:3] if s.strip())
        out.append(f"{rel}:{h.start_line}-{h.end_line} ({kind}) score={h.score:.3f}\n  {preview}")
    return "\n\n".join(out)


@server.tool()
def summarize_file(path: str) -> dict:
    """Produce a typed summary of a file: language, one_liner, public_api,
    dependencies, notable, confidence."""
    from lantern.summarize import summarize_file as sf

    full = (_REPO / path).resolve()
    if _REPO not in full.parents and full != _REPO:
        return {"error": f"path {path!r} resolves outside the repo"}
    if not full.is_file():
        return {"error": f"not a file: {path}"}
    return sf(full).model_dump()


@server.tool()
def ask(question: str, max_steps: int = 5) -> dict:
    """Run Lantern's multi-step agent on `question` against the configured
    repo. Returns the answer plus a per-step trace so the client can show
    its work."""
    from lantern.agent import agent_loop
    from lantern.trace import Trace

    trace = Trace()
    result = agent_loop(question, repo=_REPO, max_steps=max_steps, trace=trace)
    steps_summary = [
        {
            "step": i,
            "next_action": s.decision.next_action,
            "reasoning": s.decision.reasoning,
            "path": s.decision.path,
            "pattern": s.decision.pattern,
        }
        for i, s in enumerate(result.steps, 1)
    ]
    return {
        "answer": result.answer,
        "forced_final": result.forced_final,
        "run_id": result.run_id,
        "steps": steps_summary,
    }


def run() -> None:
    """Entry point used by `lantern mcp` (stdio transport)."""
    server.run()


if __name__ == "__main__":
    run()

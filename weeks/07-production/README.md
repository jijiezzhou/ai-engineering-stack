# Week 7 — Production: MCP Server + Tracing

**Goal:** turn Lantern from "a CLI on my laptop" into a tool Claude Code or Cursor can call directly. By the end:

```bash
lantern mcp                      # run as an MCP stdio server
lantern ask "..." --save-trace   # JSONL trace of every step
lantern trace                    # list recent runs
lantern trace 20260505-093412-a1b2c3   # replay one
```

…and Lantern shows up as a callable tool inside any MCP client.

## Why MCP is a big deal

Until now, Lantern was a standalone CLI. Useful, but isolated — you had to `cd` into a terminal to use it. **MCP** (Model Context Protocol) is the open standard that lets a coding assistant call third-party tools as if they were native. Claude Code, Cursor, Goose, Continue — they all speak it. Once Lantern is an MCP server, anyone using one of those clients gets your `ask`, `search`, `summarize` capabilities for free.

> Building an AI tool people use = wrap it in MCP.

## Six tools exposed

`lantern mcp` runs a stdio MCP server that exposes:

| Tool | Maps to | What an MCP client calls it for |
|---|---|---|
| `about_repo` | constant | "Which repo is Lantern serving?" |
| `read_file(path)` | week 3 | Read a file with path-traversal protection. |
| `list_dir(path)` | week 3 | Discover the project shape. |
| `grep(pattern, path)` | week 3 | Find a substring across files. |
| `search(query, top_k)` | week 5 hybrid | Semantic + BM25 retrieval, with kind/score. |
| `summarize_file(path)` | week 2 | Typed `FileSummary` (path, language, public_api, dependencies, notable, confidence). |
| `ask(question, max_steps)` | week 6 | Run the full multi-step agent. Returns answer + per-step trace. |

The repo each server exposes is set by `LANTERN_REPO` (defaults to cwd at server start). Run multiple servers if you want to expose multiple repos.

## Wiring into Claude Code

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "lantern": {
      "command": "lantern",
      "args": ["mcp"],
      "env": {
        "LANTERN_REPO": "/Users/you/Desktop/projects/ai-engineering-stack"
      }
    }
  }
}
```

Restart Claude Code. The `lantern_*` tools appear in its tool list. Ask Claude Code "explain the lantern repo" and watch it call `search` and `read_file` against your local Lantern server.

For Cursor or other MCP clients, the config shape is similar — `command` + `args` + `env`. Consult their docs for the exact path.

## Tracing: every agent run, captured

Every `agent_loop` call can take an optional `Trace`:

```python
from lantern import agent_loop, Trace

trace = Trace()
result = agent_loop("Where is RRF implemented?", repo=".", trace=trace)
print(f"saved to: {trace.path}")
```

What you get is a JSONL file under `~/.lantern/traces/<run_id>.jsonl`:

```jsonl
{"run":"20260505-093412-a1b2c3","step":-1,"ts":0.0,"kind":"run_start","question":"Where is RRF implemented?", ...}
{"run":"...","step":-1,"ts":0.4,"kind":"primer","chars":521}
{"run":"...","step":0,"ts":2.1,"kind":"decision","next_action":"read_file","reasoning":"...", ...}
{"run":"...","step":0,"ts":2.2,"kind":"tool_output","chars":1832,"preview":"..."}
{"run":"...","step":1,"ts":5.7,"kind":"answer","text":"RRF is implemented in lantern/search.py:97 ..."}
```

Why JSONL: append-only, line-oriented, greppable, plays nicely with `jq` and any future dashboard.

The CLI gives you both a list and a replay:

```bash
lantern ask "Where is RRF implemented?" --save-trace --show-trace
lantern trace                          # most-recent-first table
lantern trace 20260505-093412-a1b2c3   # pretty replay of one run
```

## What ships this week

```
lantern/
├── trace.py            ← NEW: Trace, list_runs, read_run, iter_runs
├── mcp.py              ← NEW: FastMCP server with 7 tools, stdio transport
├── agent.py            ← agent_loop accepts trace= and emits events
├── cli.py              ← +`lantern mcp`, +`lantern trace [<id>]`,
│                          +`--save-trace` flag on `ask`
└── __init__.py         ← re-exports

weeks/07-production/
├── README.md           ← this file
└── exercise.md         ← 5-min: install into Claude Code + replay traces
```

Deps: `+mcp` (the official Model Context Protocol Python SDK).

## What deliberately *didn't* ship

| Idea | Why deferred |
|---|---|
| Anthropic prompt caching (`cache_control: ephemeral`) | Doing it right requires per-message cache strategy; a bad implementation costs more than no caching. Future work. |
| Token / cost tracking | Easy to add — Anthropic returns `usage` per call. Lands when we run frontier evals in week 8. |
| Semantic answer cache (SQLite) | Useful for eval re-runs but invisible to the wow demo. |
| `eval --mode agent` | That's week 8's whole job — public benchmark across Qwen 7B / Claude / GPT. |

## Concept → Lantern mapping

| Concept | Where Lantern uses it |
|---|---|
| MCP `FastMCP` server | The "Lantern is also a service" surface, not just a CLI. |
| Stdio transport | Simplest to wire into desktop clients; no port juggling. |
| `Trace` JSONL | Week 8: every benchmark run produces traces; we'll grep them for failure modes. |
| `LANTERN_REPO` env | Lets one Lantern install serve different projects via separate Claude Code MCP entries. |

## Cost & latency notes

- The MCP server adds zero overhead — it's the same Lantern code, dispatched over stdio.
- Tracing is ~50 µs per event (one JSONL append). Negligible.
- `ask` from Claude Code takes the same time as `lantern ask` locally (the bottleneck is Qwen 7B inference, not the transport).

## Further reading

- [Model Context Protocol — spec](https://modelcontextprotocol.io/) — *the canonical reference. The server.tool decorator pattern matches our schema-first approach.*
- [Anthropic — MCP Quickstart](https://modelcontextprotocol.io/quickstart) — *15-minute walkthrough of building an MCP server in Python.*
- [Claude Code — MCP integration](https://docs.claude.com/en/docs/claude-code/mcp) — *exact paths and config formats for desktop client wiring.*

When you've done the [exercise](exercise.md), you're ready for **Week 8: polish + public benchmark** — running `eval --mode agent` across Qwen 7B, Claude Sonnet, and GPT on the same Q&A set, publishing the scoreboard.

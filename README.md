# AI Engineering Stack

> Build a local-first AI coding agent in 8 weeks — concepts you can run, an MCP server you can ship, a benchmark you can publish.

[![Stars](https://img.shields.io/github/stars/jijiezzhou/ai-engineering-stack?style=social)](../../stargazers)
[![Last commit](https://img.shields.io/github/last-commit/jijiezzhou/ai-engineering-stack)](../../commits/main)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What this is

A hands-on, eight-week path from "I can code" to **shipping a real AI engineering project**. You build **Lantern** — a local-first coding agent that:

- runs on a 16 GB Mac via Ollama (Qwen 2.5 Coder 7B), with one env var to swap to Claude;
- inspects unfamiliar codebases with `read_file`, `list_dir`, `grep`, **semantic search**, and a multi-step **agent loop with reasoning**;
- exposes itself as an **MCP server** so Claude Code or Cursor can call it directly;
- ships a **benchmark harness** that grades end-to-end agent answers via LLM-as-judge — *the* practice that separates AI engineering from AI tinkering.

Not an awesome-list. Not a theory dump. **Clone → follow → ship.**

## Status

✅ **Curriculum complete — all 8 weeks shipped.** See [BENCHMARKS.md](BENCHMARKS.md) for the published results.

## Quick start

```bash
# 1. Install Ollama and pull the default models (~5 GB total)
brew install ollama
ollama serve &
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Clone and sync
git clone https://github.com/jijiezzhou/ai-engineering-stack
cd ai-engineering-stack
uv sync

# 4. Try every week's capability in 60 seconds
uv run lantern chat "Reply with one word: ready" -t 0.0          # week 1
uv run lantern summarize lantern/llm.py                          # week 2
uv run lantern ask "Where is path traversal blocked?" --show-trace  # week 3+6
uv run lantern index .                                           # week 4
uv run lantern search "the brain that decides which tool to call" -k 3  # week 4+5
uv run lantern eval --mode retrieval --skip-rerank               # week 5
uv run lantern eval --mode agent --questions 4                   # week 8
LANTERN_REPO=$(pwd) uv run lantern mcp                           # week 7 — runs the MCP server
```

Install globally so you can drop the `uv run` prefix from anywhere:

```bash
uv tool install --editable .
uv tool update-shell        # one-time PATH fix
# new terminal, then:
lantern ask "What does this package expose?" --show-trace
```

Frontier swap (when you want better quality):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
LANTERN_BACKEND=anthropic lantern ask "Trace the flow from CLI to ToolSpec.run"
```

## The 8-week path

The capstone is **Lantern** — point it at any repo, it answers questions about the code by inspecting files. Each week ships one slice.

| Week | You learn | You build | Lantern slice |
|-----:|-----------|-----------|---------------|
| 1 | LLM fundamentals + local-model setup | Streaming CLI talking to Qwen 2.5 Coder | Core `LLM` client (local + frontier) |
| 2 | Structured output with Pydantic | Typed `summarize_file` | Output contracts + retry on validation |
| 3 | Tool use & function calling | `read_file` / `list_dir` / `grep` w/ path-traversal protection | Tool layer |
| 4 | Embeddings + tree-sitter chunking | `lantern index .` + semantic `search` | Retrieval v1 (Chroma + nomic-embed-text) |
| 5 | Hybrid retrieval + reranking + **evals** | BM25 sidecar, RRF, LLM rerank, golden Q&A harness | Retrieval v2 + first eval table |
| 6 | Agents: planning, memory, guardrails | Multi-step `ask` with reasoning + retrieval primer | Agent loop with `--show-trace` |
| 7 | Production: MCP + tracing | `lantern mcp` (callable from Claude Code), JSONL traces | The "anyone can use this" surface |
| 8 | Polish + public benchmark | `eval --mode agent` with LLM-as-judge, [BENCHMARKS.md](BENCHMARKS.md) | The artifact people will star |

Each week's folder (`weeks/NN-name/`) contains: concept walkthrough, runnable code, hands-on exercise, and a checkpoint that plugs into Lantern.

## What you'll have at the end

- **A 1,500-line, fully readable Python package** that wraps every modern AI engineering pattern: structured output, tool calling, RAG, hybrid retrieval, reranking, agents, MCP, evals.
- **An MCP server** in `~/.local/bin/lantern` that Claude Code or Cursor can use to explore *any* repo on your machine — yours, your team's, an open-source clone.
- **A published benchmark** (`BENCHMARKS.md`) on a known test set, reproducible in 30 minutes. Hiring managers have something to read.
- **The skill of measuring AI quality.** This is the rarest capability in the field. After week 8, you debate retrieval and agent quality with numbers, not vibes.

## Layout

```
.
├── README.md
├── BENCHMARKS.md                    ← published numbers + methodology
├── pyproject.toml
├── lantern/                         ← the cumulative capstone
│   ├── llm.py        chat / structured / call    (weeks 1, 2, 3)
│   ├── summarize.py  FileSummary + summarize_file (week 2)
│   ├── tools.py      ToolSpec + 3 safe tools      (week 3)
│   ├── chunk.py      tree-sitter + fixed chunking (week 4)
│   ├── index.py      Chroma + BM25 indexing       (weeks 4, 5)
│   ├── search.py     vector / bm25 / hybrid       (weeks 4, 5)
│   ├── rerank.py     LLM-as-judge rerank          (week 5)
│   ├── evals.py      retrieval + agent eval       (weeks 5, 8)
│   ├── agent.py      single + multi-step          (weeks 3, 6, 7)
│   ├── trace.py      JSONL trace                  (week 7)
│   ├── mcp.py        FastMCP stdio server         (week 7)
│   └── cli.py        Typer entry point
├── weeks/
│   ├── 01-foundations/
│   ├── 02-structured-output/
│   ├── 03-tool-use/
│   ├── 04-embeddings/
│   ├── 05-hybrid-evals/
│   ├── 06-agents/
│   ├── 07-production/
│   └── 08-benchmark/
└── evals/lantern.yaml               ← 16 golden Q&A
```

## Why local-first

- **Runs on a 16 GB Mac.** Qwen 2.5 Coder 7B via Ollama, ~5 GB RAM, ~40 tok/s on Apple Silicon.
- **No data leaves your machine.** Paste your private repo without telemetry concerns.
- **Free to learn.** No API keys, no per-token costs while you experiment.
- **Frontier-optional.** One env-var swap to Claude when you want quality.
- **Forces good engineering.** Smaller models won't hide weak prompts or missing evals — you have to actually measure.

## Curation principles

- **Build, don't just read.** Every section ships code you run in under 5 minutes.
- **One project, eight slices.** Sections compose — week N's output is week N+1's input.
- **Production lens.** Cost, latency, evals, and failure modes are first-class, not appendices.
- **Local-first, frontier-optional.** Defaults are open; quality upgrades are one env-var away.
- **Honest pedagogy.** When the demo reveals reality is worse than the rosy plan, ship reality and frame the gap as the lesson. (See [BENCHMARKS.md](BENCHMARKS.md) — the local-7B Correctness number is brutal, and that's the point.)

## Contributing

PRs welcome — clearer explanations, better exercises, bug fixes in week code, and especially:
- a real `bge-reranker` swap for week 5;
- index-time chunk-type filtering to fix the "docs over source" problem;
- additional eval test sets for popular OSS repos (FastAPI, Express, etc.);
- backend support for OpenAI / Gemini in `lantern/llm.py`.

New external links must include a one-line *why it's here* note. Open an issue first for roadmap changes.

## License

MIT

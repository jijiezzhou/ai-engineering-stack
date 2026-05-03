# AI Engineering Stack

> Build a local-first AI coding agent in 8 weeks — concepts you can run, not just read.

[![Stars](https://img.shields.io/github/stars/jijiezzhou/ai-engineering-stack?style=social)](../../stargazers)
[![Last commit](https://img.shields.io/github/last-commit/jijiezzhou/ai-engineering-stack)](../../commits/main)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What this is

A hands-on path from "I can code" to **shipping a real AI engineering project**. You'll build **Lantern** — a local-first coding agent that explains unfamiliar codebases like a new hire would, ships with an open eval harness, and plugs into Claude Code or Cursor as an MCP server.

Runs on a 16 GB Mac. No API keys required (frontier models optional).

Not an awesome-list. Not a theory dump. **Clone → follow → ship.**

## Status

🚧 **Week 4 of 8.** Lantern can semantically search a codebase — find the right function by intent, not just by literal name. Weeks 5–8 land week-by-week. Star to follow along.

## Quick start

```bash
# 1. Install Ollama and pull the default model (~5 GB)
brew install ollama   # or https://ollama.com/download
ollama serve &        # in another terminal, or run the Ollama app
ollama pull qwen2.5-coder:7b

# 2. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Sync deps
uv sync

# 4. Talk to your local LLM
uv run lantern chat "Explain Python decorators in 3 lines"

# 5. Get a structured summary of any source file
uv run lantern summarize lantern/llm.py

# 6. Ask a question — Lantern picks a tool, reads the code, then answers
uv run lantern ask "What does this package expose?"

# 7. Semantic search (index once, search forever)
ollama pull nomic-embed-text          # ~140 MB, one time
uv run lantern index .
uv run lantern search "where is path traversal blocked" -k 3
```

Install globally (optional — drop the `uv run` prefix from any directory):
```bash
uv tool install --editable .
uv tool update-shell   # one-time: adds ~/.local/bin to your PATH
# open a new terminal, then:
lantern chat "Hello"
lantern summarize ~/anywhere/file.py
```
The `--editable` flag links the install back to the source — your week 3+ changes are picked up live, no reinstall needed.

Frontier swap (optional, when you want better quality):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
LANTERN_BACKEND=anthropic uv run lantern chat "Hello"
```

## The Build Path

The capstone is **Lantern**: point it at a repo, it indexes the code, answers questions, and explains modules — like a senior engineer onboarding you. Each week ships one slice.

| Week | You learn | You build | Lantern slice |
|-----:|-----------|-----------|---------------|
| 1 | LLM fundamentals + local-model setup | Streaming CLI talking to Qwen2.5-Coder | Core LLM client (local + frontier) |
| 2 | Structured output with Pydantic | Typed file-summarizer | Output contracts |
| 3 | Tool use & function calling | Agent w/ `read_file`, `list_dir`, `grep` | Tool layer |
| 4 | Embeddings + code-aware chunking (tree-sitter) | Semantic code search v1 | Retrieval v1 |
| 5 | Hybrid retrieval + reranking + **evals** | Golden Q&A harness on a real OSS repo | Retrieval v2 + eval harness |
| 6 | Agents: planning, memory, guardrails | The onboarding agent loop | Orchestration |
| 7 | Caching, traces, **MCP server** | Lantern as an MCP server (use it from Claude Code/Cursor) | Production |
| 8 | Polish + public benchmark | Qwen 7B vs Claude vs GPT scoreboard | Ship 🚀 |

Each week's folder contains: concept walkthrough, runnable code, hands-on exercise, and a checkpoint that plugs into Lantern.

## Layout

```
.
├── README.md
├── pyproject.toml
├── lantern/             # the cumulative capstone — grows weekly
│   └── llm.py           # week 1: unified LLM client
├── weeks/
│   ├── 01-foundations/  # ← start here
│   ├── 02-structured-output/
│   └── ...              # one folder per week, each runnable standalone
└── resources/           # only the links that earned their spot
```

*(Folders land as each week ships.)*

## Why local-first

- **Runs on a 16 GB Mac.** Qwen2.5-Coder-7B via Ollama is ~5 GB RAM, ~40 tok/s on Apple Silicon.
- **No data leaves your machine.** Paste your private repo without telemetry concerns.
- **Free to learn.** No API keys, no per-token costs while you experiment.
- **Frontier-optional.** One env-var swap to Claude/GPT when you want quality.
- **Forces good engineering.** Smaller models won't hide weak prompts or missing evals — you have to actually measure.

## Curation principles

- **Build, don't just read.** Every section ships code you run in under 5 minutes.
- **One project, eight slices.** Sections compose — week N's output is week N+1's input.
- **Production lens.** Cost, latency, evals, and failure modes are first-class, not appendices.
- **Local-first, frontier-optional.** Defaults are open; quality upgrades are one env-var away.
- **Depth > breadth.** One great resource per topic, with a one-line *why it's here*.

## Contributing

PRs welcome — clearer explanations, better exercises, bug fixes in week code. New external links must include the one-line *why it's here*. Open an issue first for roadmap changes.

## License

MIT

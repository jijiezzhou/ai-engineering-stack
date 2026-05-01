# AI Engineering Stack

> Build a production-grade AI app in 8 weeks — one section at a time. Concepts you can run, not just read.

[![Stars](https://img.shields.io/github/stars/zjzhou/ai-engineering-stack?style=social)](../../stargazers)
[![Last commit](https://img.shields.io/github/last-commit/zjzhou/ai-engineering-stack)](../../commits/main)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What this is

A hands-on path from "I can code" to **shipping a real AI product**. Each week pairs a core concept with code you run locally, and every section adds a piece to one cumulative **capstone project** — by week 8 you have a deployed, evaluated, observable AI app in your portfolio.

Not an awesome-list. Not a theory dump. **Clone → follow → ship.**

## Status

🚧 **Week 0 — scaffolding.** Roadmap locked, content rolling in week-by-week. Star to follow along.

## The Build Path

The capstone is a **personal knowledge assistant**: a RAG-powered agent that answers questions over your own documents, calls tools, and runs in production with evals and tracing. Each week ships one slice.

| Week | You learn | You build | Plugs into capstone as |
|-----:|-----------|-----------|------------------------|
| 1 | LLM fundamentals: tokens, context, sampling | A prompt-driven CLI that talks to an LLM | Core LLM client + config |
| 2 | Prompt engineering & structured output | Schema-validated extractor (JSON out) | Output contract for the agent |
| 3 | Tool use & function calling | Agent that calls real APIs (search, calc) | Tool layer |
| 4 | Embeddings & vector search | "Chat with your notes" v1 | Retrieval layer |
| 5 | Production RAG: hybrid search + reranking | Q&A with citations + eval harness | Retrieval v2 + eval suite |
| 6 | Agents: planning, memory, guardrails | Multi-step task agent | Orchestration |
| 7 | Productionization | Deployed web app + tracing + caching | Deploy + observability |
| 8 | Capstone polish | Portfolio writeup, demo, post-mortem | Ship 🚀 |

Each week's folder contains: `README.md` (concept + walkthrough), runnable code, a checkpoint test, and a 5-minute exercise.

## Layout

```
.
├── README.md
├── CLAUDE.md
├── weeks/
│   ├── 01-llm-fundamentals/
│   ├── 02-prompting/
│   └── ...                  # one folder per week, each runnable standalone
├── capstone/                # the cumulative project, grown week-by-week
└── resources/               # only the links that earned their spot
```

*(Folders land as each week ships.)*

## Curation principles

- **Build, don't just read.** Every section ships code you can run in under 5 minutes.
- **One project, eight slices.** Sections compose — week N's output is week N+1's input.
- **Production lens.** Cost, latency, evals, and failure modes are first-class, not appendices.
- **Depth > breadth.** One great resource per topic, with a one-line *why it's here*.
- **No vendor worship.** Tools change; the patterns don't.

## Contributing

PRs welcome — especially: clearer explanations, better exercises, bug fixes in week code. New external links must include the one-line *why it's here*. Open an issue first for roadmap changes.

## License

MIT

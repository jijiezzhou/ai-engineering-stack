# Changelog

The 10-week curriculum, week by week, with the slice each one added to
`lantern/`. Quantitative results live in [BENCHMARKS.md](BENCHMARKS.md);
this file is the narrative.

The format is intentionally chronological — readers can scroll from week 1
to week 10 and see the project grow.

## Week 10 (extension) — Production ops

Commit: [`80d8ec7`](../../commit/80d8ec7)

Agent loop hardening + CI eval gate. Acts on the synthesis bottleneck the
week-8 benchmark exposed: the 7B agent was finding the right files but
looping on identical grep calls and writing hedged answers.

- **Tool-call dedup** with cached output — collapses "grep-grep-grep"
  loops; doesn't cache error outputs so the model can retry.
- **Validate-and-retry on `Decision`** — catches the Qwen-7B failure
  where `next_action="grep"` ships with `pattern=""`. Re-prompts with
  the specific validation error fed back.
- **Two-stage forced-final** — summarize what each step revealed, then
  commit. Makes forced answers specific instead of hedged.
- **`max_steps` default 5 → 8** — cheap rope.
- **CI eval gate** at `.github/workflows/eval.yml` — retrieval regression
  check on every PR.

Result on 4 questions: Correctness **0.25 → 0.50**, File-hit **0.50 → 1.00**.

## Week 9 (extension) — Production retrieval

Commit: [`ef4259b`](../../commit/ef4259b)

Chunk-type filtering. Acts on the week-5/8 finding: docs that *describe*
a symbol outrank source that *defines* it on a mixed corpus.

- Every chunk now carries a `chunk_class` field (`code | doc | config | other`),
  set at index time by file suffix.
- All three retrievers (`search`, `bm25_search`, `hybrid_search`) accept
  an optional `kinds=` filter.
- The agent's retrieval primer defaults to `kinds=["code"]`.

Result: hybrid R@5 0.62 → **1.00**; BM25 R@1 0.00 → **0.69**; agent
Correctness 0.00 → **0.25**.

## Week 8 — Public benchmark

Commit: [`f4a8590`](../../commit/f4a8590)

End-to-end agent evaluation with LLM-as-judge. The first quantitative
quality measurement in the curriculum.

- `TestCase` gains `golden_answer`; every test case in `evals/lantern.yaml`
  now has a one-paragraph human-written reference.
- `evaluate_agent()` runs `agent_loop` per question and grades via
  `LLM.structured()` (same pattern as week-5 rerank).
- `lantern eval --mode agent` ships with `--max-steps`, `--questions`,
  `--judge-backend`.
- `BENCHMARKS.md` published — methodology, real numbers, reproduction
  instructions.

## Week 7 — Production: MCP server + tracing

Commit: [`adaa977`](../../commit/adaa977)

Lantern becomes callable from any MCP-aware client.

- `lantern/mcp.py` — FastMCP stdio server exposing 7 tools
  (`about_repo`, `read_file`, `list_dir`, `grep`, `search`,
  `summarize_file`, `ask`). Operates on `LANTERN_REPO`.
- `lantern/trace.py` — `Trace` class that writes JSONL events for every
  agent run to `~/.lantern/traces/<run_id>.jsonl`.
- CLI: `lantern mcp` and `lantern trace [<id>]`. `--save-trace` on `ask`.

## Week 6 — Multi-step agent loop

Commit: [`d804785`](../../commit/d804785)

The week-3 two-shot becomes a real loop.

- `Decision.reasoning` field — one-sentence think-before-act.
- Retrieval primer at step 0 (week 5 → week 6 pipeline).
- Step-aware memory with capped tool output.
- `max_steps` guardrail with structured forced-final.
- `lantern ask` is multi-step by default; `--single-step` preserves
  week 3 behavior; `--show-trace` prints the agent's work.

## Week 5 — Hybrid retrieval + rerank + evals

Commit: [`253b263`](../../commit/253b263)

The first time AI quality stops being vibes and becomes a number.

- `BM25Index` (sparse retrieval, `rank_bm25`).
- `hybrid_search()` — Reciprocal Rank Fusion of vector + BM25.
- `rerank()` — LLM-judged passage relevance via `LLM.structured()`.
- `evaluate()` harness with Recall@k and MRR.
- 16-question golden test set at `evals/lantern.yaml`.
- CLI: `lantern eval --mode retrieval`.

## Week 4 — Embeddings + tree-sitter chunking

Commit: [`7a26888`](../../commit/7a26888)

Semantic search over the codebase.

- `chunk_file()` — tree-sitter for Python (one chunk per top-level
  def/class, plus a header chunk), fixed-size for everything else.
  `_cap()` re-splits oversized semantic chunks.
- `index_repo()` — embeds via `nomic-embed-text` through Ollama,
  persists to Chroma at `~/.lantern/index/<repo>/`.
- `search()` — top-k cosine over the dense index.
- CLI: `lantern index <repo>` and `lantern search <query>`.

## Week 3 — Tool use & function calling

Commit: [`f83bef1`](../../commit/f83bef1)

The agent gets eyes.

- `ToolSpec` Pydantic base — class name → tool name, docstring →
  description, fields → args, `run()` → executor.
- Three tools: `read_file`, `list_dir`, `grep`. All path-traversal-safe
  via `_resolve_safely`. Output truncated.
- Single-step `ask()` orchestrator. Uses `LLM.structured()` with a
  `Decision` schema rather than native tool calling — Qwen 7B's native
  channel is unreliable.
- CLI: `lantern ask "..." --repo <path>`.

## Week 2 — Structured output

Commit: [`3e046b6`](../../commit/3e046b6)

Free-form text becomes Pydantic.

- `LLM.structured(prompt, schema)` — Ollama `format=<json-schema>` /
  Anthropic tool-use, with one retry on validation error.
- `FileSummary` + `summarize_file()` — typed structured summary of any
  source file.
- CLI: `lantern summarize <file>` (with `--json` for pipe-to-jq).

## Week 1 — LLM fundamentals

Commit: [`1565fa0`](../../commit/1565fa0)

The foundation. Tokens, context, sampling, streaming.

- `LLM` class wrapping Ollama (local default: `qwen2.5-coder:7b`) and
  Anthropic (`claude-sonnet-4-6`) behind one streaming interface.
- `.stream()`, `.complete()` methods.
- CLI: `lantern chat "..."` with `--temperature` and `--system`.
- `pyproject.toml` with `uv`/Hatchling build.
- MIT license, `.gitignore`, README with the 8-week pitch.

## Week 0 — Scaffolding

Commits: [`8743483`](../../commit/8743483), [`6817c49`](../../commit/6817c49)

Initial README, repo layout, GitHub badge URLs. The 8-week pitch.

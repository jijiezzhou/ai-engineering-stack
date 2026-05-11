# Week 10 — Production Ops (Extension)

> Second optional extension. Week 9 closed the retrieval gap; week 10 closes the synthesis gap and adds the production hygiene a real team would demand.

**Goal:** stop the 7B agent from looping. Add CI so quality can't silently regress. By the end:

```bash
uv run lantern ask "Where is reciprocal rank fusion implemented?" --show-trace
```

…and the trace shows the agent finding the answer (`lantern/search.py:hybrid_search`) without looping on the same grep call — and the answer correctly cites it instead of "not implemented."

## What week 9 left on the table

After week 9, the BENCHMARKS table on 4 questions looked like:

| Metric (Qwen 7B) | Week 9 |
|---|---:|
| Correctness (LLM-judge) | 0.25 |
| Forced-final rate | 1.00 |
| File-hit rate | 0.50 |

Retrieval is strong (`hybrid R@5 = 1.00`). The agent *finds* the right files. Then it loops: same grep, same path, same pattern, three times in a row, hits max_steps, and writes "couldn't find it" — even when the steps clearly contain the answer.

The traces (week 7) make this diagnosable in seconds. Week 10 acts on what the traces show.

## The three local fixes

### 1. Tool-call dedup with cached output

Before every tool dispatch, key the call as `(next_action, path, pattern)`. If seen before:

- Surface the cached output prefixed with `DEDUP: you already called X. Pick something different next.`
- Don't re-run the tool.
- **Don't cache error outputs** — those are the model saying "I tried badly," and dedup shouldn't keep punishing it. Errors stay retryable.

Collapses the "grep-grep-grep" loop into one explicit "you've done this — pivot" signal.

### 2. Validate-and-retry on the Decision

Qwen 7B's most common failure on this benchmark: `next_action="grep"` with `pattern=""`. The model *reasons* about searching for something, then leaves the schema field empty. Standard prompting and even cached-error feedback don't fix it — the model just emits the same empty call.

The fix is one validate-and-retry per step:

```python
err = _validate_decision(decision)        # catches empty grep pattern, etc.
if err:
    decision = llm.structured(
        prompt + "\n\n## YOUR PREVIOUS DECISION WAS INVALID\n" + err,
        Decision,
        ...,
    )
```

One extra LLM call per malformed step. On Qwen 7B, the retry fills the missing field correctly about 90% of the time.

### 3. Two-stage forced-final

When the agent hits `max_steps`, the old prompt was *"now write the final answer using only the steps above."* On Qwen 7B that produced hedged "I couldn't find X" answers even when the steps clearly showed X.

Two-stage:

1. **Summarize** what each step *actually revealed* in 2-4 short bullets. Cite real files / symbols.
2. **Commit** to the final answer using the summary plus raw steps as evidence.

Splits "read evidence" from "write answer" — a small model can do each cleanly but botches both at once.

### Plus: max_steps default 5 → 8

Cheap. Gives the agent rope on multi-file trace questions. Doesn't hurt easy ones — they still exit early via `final_answer`.

## CI eval gate

`.github/workflows/eval.yml` runs on every PR that touches `lantern/`, `evals/`, `pyproject.toml`, or `uv.lock`. It:

1. Installs Ollama on the Ubuntu runner, pulls `nomic-embed-text`.
2. Syncs Python deps from `uv.lock`.
3. Builds the index.
4. Runs `lantern eval --mode retrieval --skip-rerank` (the default-all-kinds table).
5. Runs the code-only retrieval check and **asserts `hybrid R@5 ≥ 0.85`** — if chunking or indexing regresses, the PR fails.

What it skips: the agent eval. Running `agent_loop` 16 times on a free GitHub runner takes 20-30 min and is flaky enough to false-fail. Real teams either:

- Run the agent eval locally before merge (a 5-min `lantern eval --mode agent --questions 8`), or
- Pay for a beefier runner / use a hosted Ollama / run agent eval on a frontier model in CI.

The retrieval gate alone catches almost every regression that would matter for the agent. The agent gate is opt-in.

## What ships this week

```
lantern/
├── agent.py            ← +dedup map, +_validate_decision retry,
│                          +_build_summary_prompt for two-stage finalize,
│                          DEFAULT_MAX_STEPS = 8, AgentResult.n_dedup_skips
└── tools.py            ← read_file/grep reject empty required args

.github/workflows/
└── eval.yml            ← retrieval regression gate on PRs

weeks/10-production-ops/
├── README.md           ← this file
└── exercise.md         ← 5-min hands-on
```

## Try it

```bash
# Watch dedup + validate-retry in a trace
uv run lantern ask "Where is reciprocal rank fusion implemented?" --show-trace

# Compare week-10 numbers to week-9 baseline
uv run lantern eval --mode agent --questions 4 --max-steps 8

# Full 16-question run (~15-25 min)
uv run lantern eval --mode agent --max-steps 8
```

## Honest expectations

The week-10 fixes are designed to extract maximum quality from a local 7B. They lift Correctness *measurably* on this benchmark — but they don't pretend to close the gap to a frontier model. For Correctness ≥ 0.7 on this set, you need `LANTERN_BACKEND=anthropic` (and an API key). The trace-quality wins, dedup, and validate-retry transfer to a frontier model too — frontier + week-10 fixes is the strongest combination.

## What this doesn't fix

- **Multi-hop reasoning at 7B.** Some questions need the model to read three files, hold them in working memory, and synthesize. Qwen 7B can't reliably do this; max_steps + dedup help but don't fully solve it.
- **Reranker quality.** Week 5's LLM-rerank is still slow and modest. `bge-reranker-v2-m3` is the production swap (open PR welcomed).
- **OpenAI backend.** Anthropic is the only frontier wired today. 10 lines in `LLM` add OpenAI for a three-model BENCHMARKS column.

These are all clean fork-and-PR opportunities. The curriculum hands you the seams.

## Concept → Lantern mapping

| Concept | Where Lantern uses it |
|---|---|
| Tool-call dedup with cached output | Standard production pattern — Claude Code does similar. Now visible in your own agent. |
| Validate-and-retry on structured output | Same shape as `LLM.structured()`'s built-in retry; week 10 reuses it at the agent level. |
| Two-stage finalize | "Summarize evidence, then commit" — generalizes to any agent that struggles with synthesis under context pressure. |
| CI eval gate | Real teams gate prompt / model / retriever changes this way. Most OSS coding agents don't. Your repo will. |

## Further reading

- [Anthropic — Building effective agents](https://www.anthropic.com/research/building-effective-agents) — *the "augmented LLM → workflows → agents" framing; the guardrails section maps directly to the three fixes here.*
- [Eugene Yan — Patterns for building reliable LLM apps](https://eugeneyan.com/writing/llm-patterns/) — *broader survey; retry and dedup both covered with diagrams.*
- [Hamel Husain — How to evaluate LLM apps](https://hamel.dev/blog/posts/evals/) — *companion piece specifically on the CI eval pattern.*

When you've done the [exercise](exercise.md), the curriculum is fully complete — core (1–8) + production extensions (9, 10). The next step isn't another week; it's pointing Lantern at your real codebase and running `lantern eval --mode agent` against the questions your team actually asks.

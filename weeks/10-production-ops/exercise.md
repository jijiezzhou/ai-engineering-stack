# Week 10 Exercise (10–20 min)

Feel each of the three local fixes in isolation, then together. Then wire up CI.

## 1. Watch the dedup guard fire

Pick a question the week-8 agent looped on:

```bash
uv run lantern ask "Where is reciprocal rank fusion implemented?" --show-trace --max-steps 6
```

In the trace, look for `output:` lines starting with `DEDUP: you already called ...`. That's a step where the model emitted a tool call identical to a previous step — the cached result is surfaced instead of re-running. **The step still counts against the budget**, but the prompt now explicitly says "pick something different."

## 2. Watch the validate-retry catch an empty grep

```bash
uv run lantern ask "Where is BM25 sparse retrieval implemented?" --show-trace --max-steps 6
```

In one of the early steps, Qwen 7B often emits `next_action=grep` with `pattern=""`. With week-10 validation:

- The agent rejects that decision *before* running the tool.
- It re-prompts with `"Your previous decision was invalid: next_action='grep' but the pattern field is empty..."`.
- The model usually fills `pattern="BM25"` and the search works.

You won't see the retry in `--show-trace` (it happens inside one step). Save the JSONL trace to see it:

```bash
uv run lantern ask "Where is BM25 sparse retrieval implemented?" --save-trace --max-steps 6
LATEST=$(ls -t ~/.lantern/traces/*.jsonl | head -1)
jq '.kind' "$LATEST" | sort | uniq -c
```

Sometimes the count of `decision` events exceeds the count of `tool_output` events — those are validate-retries.

## 3. Watch the two-stage finalize

Force a max-steps hit:

```bash
uv run lantern ask "Explain every public symbol in lantern, file by file." --show-trace --max-steps 3
```

When the budget exhausts, the trace shows a `summary` event (the model's bullet-point digest of what it saw) before the `forced_final` event (the committed answer). Compare the final answer's specificity to what week 8 produced on the same question — the week-10 answer cites more specific files.

## 4. Run the agent eval and compare to week-9 numbers

```bash
uv run lantern eval --mode agent --questions 4 --max-steps 8
```

Compare to the week-9 row in [BENCHMARKS.md](../../BENCHMARKS.md). Correctness should be visibly higher; forced-final rate often drops. Run twice if the numbers look noisy — small model + small N = real variance.

## 5. Inspect the CI gate

Open `.github/workflows/eval.yml`. It runs on PRs that touch `lantern/`, `evals/`, or deps. Steps:

1. Install Ollama, pull `nomic-embed-text`.
2. `uv sync --frozen` (uses your locked deps).
3. `lantern index . --rebuild`.
4. `lantern eval --mode retrieval --skip-rerank` (default-all-kinds table).
5. `assert hybrid R@5 ≥ 0.85 with kinds=["code"]` — the regression gate.

Open a PR with a no-op change in `lantern/` to see it run on your fork.

## 6. Stretch — break the gate intentionally

Make a one-line edit that breaks chunking quality (e.g. in `lantern/chunk.py` change `_chunk_python` to always return `[]`, forcing fallback to fixed-size for everything). Open a PR. The CI run should:

- Build the index successfully (fixed-size chunking still produces chunks).
- Run retrieval eval — vector/bm25 numbers will drop.
- Hit the `R@5 ≥ 0.85` assert and **fail the workflow**.

Revert the change. The PR goes green. **Now you know the CI gate is actually doing work.**

## 7. Frontier comparison (needs ANTHROPIC_API_KEY)

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Frontier agent + Qwen judge — isolates agent quality
LANTERN_BACKEND=anthropic uv run lantern eval --mode agent --questions 4

# Frontier judge too — most credible publication setup
LANTERN_BACKEND=anthropic uv run lantern eval --mode agent --questions 4 --judge-backend anthropic
```

Expected Claude Correctness on this set: ~0.75 in ~5-10 s per question (vs Qwen's 0.25-0.50 in 30-60 s). **That's the gap a real production team is paying for** when they pick a frontier model.

## 8. Run on your team's repo with CI

```bash
cd ~/work-repo
uv run lantern index . --rebuild
$EDITOR evals/work.yaml          # 10-20 onboarding questions, each with golden_answer
uv run lantern eval --mode agent --tests evals/work.yaml --repo . --questions 6
```

Copy `.github/workflows/eval.yml` into your work-repo, adjust the test file path, commit. Now every PR runs the eval gate. **A junior engineer changing a prompt sees the Correctness delta in the PR comment.** That's the actual ops loop of a real AI engineering team in 2026.

---

When you've done these, you've shipped:
1. A local agent that loops less and finalizes more confidently.
2. Honest before/after benchmark numbers.
3. CI infrastructure that prevents silent quality regression.

The 10-week curriculum is complete. Everything past this is forking and shipping — `bge-reranker`, OpenAI backend, prompt caching, a real product on top of these primitives. PRs welcome.

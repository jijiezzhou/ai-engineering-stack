# Week 8 Exercise (~10–30 min, mostly waiting)

The agent eval is slow on a 7B local model. Plan for ~1 minute per question on Qwen 7B + ~5 s for the judge. Set a coffee timer.

## 1. Fast pass — sanity check on 4 questions

```bash
uv run lantern eval --mode agent --questions 4
```

Expect ~3-6 minutes. You'll see one line per question:

```
  ✓   31s 3st  conf=0.95  Where is BM25 sparse retrieval implemented?
  ✗   58s 4st* conf=0.40  Where is reciprocal rank fusion applied?
```

`*` after the step count means the run was forced to final at max_steps.

## 2. Full local pass — all 16 questions

```bash
uv run lantern eval --mode agent
```

Walk away for 15-30 minutes. When you come back, read the table.

Three things to check:
- **Correctness**: 0.5 to 0.7 is typical for Qwen 7B on this set.
- **File-hit rate**: should be >0.7 — retrieval is doing its job.
- **Forced-final rate**: anything above 0.3 means the agent is looping. Trace one of those (`--save-trace` on `ask`) and read the per-step reasoning.

## 3. The credibility upgrade — frontier judge

If you have an Anthropic API key, re-run with Claude as the judge. Same agent (Qwen), different grader:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run lantern eval --mode agent --questions 4 --judge-backend anthropic
```

Compare the Correctness number to your earlier 4-question run. Same agent answers; different grader. Often the frontier judge is **stricter** — you'll see a few `correct=true` flips to `false`. **That's the mechanism the published number in BENCHMARKS.md uses.**

## 4. Compare two agents end-to-end

```bash
# Local agent + local judge (cheap)
uv run lantern eval --mode agent --questions 4 > /tmp/local.txt

# Frontier agent + frontier judge (expensive but the benchmark's gold)
LANTERN_BACKEND=anthropic uv run lantern eval --mode agent --questions 4 --judge-backend anthropic > /tmp/frontier.txt

diff <(grep "│" /tmp/local.txt) <(grep "│" /tmp/frontier.txt)
```

This is the benchmark in `BENCHMARKS.md` reduced to two questions. Now you understand exactly what the published numbers measure.

## 5. Read a failure trace

Find a question where Qwen got `✗` in step 1. Re-run that question with full trace:

```bash
uv run lantern ask "<the question text>" --save-trace --show-trace
uv run lantern trace <run_id>
```

In the trace, look for:
- Did the agent open any of the expected files?
- Did it loop on the same tool call?
- Did the final answer say "not implemented" when the reference says it IS implemented?

> Reading these traces is the closest most engineers get to "watching how a teammate thinks." It's also where almost every meaningful prompt improvement comes from.

## 6. Run the benchmark on YOUR codebase

```bash
# Pick any cloned repo
cd /path/to/your-real-repo
uv run lantern index .

# Write 5-10 questions you'd actually ask a new hire about this code
$EDITOR evals/your-repo.yaml

# Run
uv run lantern eval --mode agent --tests evals/your-repo.yaml --repo /path/to/your-real-repo
```

This is what `lantern eval --mode agent` is *for*. The Lantern repo is just a self-test. The real value is when you point this at your team's codebase and use the resulting numbers to argue for budget — "our agent answers 64% of onboarding questions correctly today; with bge-reranker swapped in, it gets to 78%."

---

**You're done.** You built a local-first AI coding agent with retrieval, multi-step reasoning, MCP server exposure, and a quantitative benchmark. That puts you ahead of most "AI engineers" in 2026. The repo on your disk is your portfolio — point hiring managers at the README, the benchmark, and the per-week walkthroughs.

If you want to keep going: real production swaps (bge-reranker, prompt caching, semantic answer cache, OpenAI backend, JSON Schema-validated MCP outputs) all hang off the seams the curriculum left open. PRs welcome.

# Lantern Benchmarks

End-to-end quality numbers for Lantern, measured by `lantern eval` against a 16-question golden test set. **Reproduce on your machine in 10–30 minutes** — every command in this file runs locally.

## What's measured

Two harnesses, one test set (`evals/lantern.yaml`).

### Retrieval (week 5)

`lantern eval --mode retrieval` runs each question through four retrievers and scores by `Recall@k` (did any of the question's `expected_paths` appear in the top-k hits?) and `MRR` (mean reciprocal rank of the first expected hit).

### Agent end-to-end (week 8)

`lantern eval --mode agent` runs the full multi-step `agent_loop` for each question, then grades the agent's final answer against a one-paragraph human-written `golden_answer` via LLM-as-judge. Reports:

- **Correctness** — fraction of answers the judge marked correct.
- **File-hit rate** — fraction of questions where any expected path appeared in the agent's tool args, tool outputs, or final answer. Decouples retrieval quality from synthesis quality.
- **Avg steps** — mean number of decisions per question (5-step budget + 1 forced-final = 6 max).
- **Forced-final rate** — fraction of runs that hit max_steps without converging. High = the agent is looping.
- **Total elapsed** — wall time, the cost-of-evals signal.

## The numbers (this Mac, 2026-05-09)

Hardware: MacBook Air 15" M4, 16 GB unified memory.
Models: `qwen2.5-coder:7b` via Ollama for both agent and judge in this run.

### Retrieval

`lantern eval --mode retrieval --skip-rerank` over all 16 questions, top-5:

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR  | Time  |
|-----------|---------:|---------:|---------:|-----:|------:|
| vector    |     0.31 |     0.69 |     0.81 | 0.49 |  0.8s |
| bm25      |     0.00 |     0.44 |     0.62 | 0.22 |  0.0s |
| hybrid    |     0.31 |     0.69 |     0.75 | 0.48 |  0.4s |
| hybrid+rerank* | 0.31 | 0.69 | 0.81 | 0.49 | ~12 min |

\* rerank is full-corpus LLM judging — slow on a 7B local model. The number listed is from a separate full run; `--skip-rerank` is the default for iteration speed.

### Agent end-to-end (Qwen 7B agent + Qwen 7B judge)

`lantern eval --mode agent --questions 4 --max-steps 5` (4 questions for benchmark publication; full 16 takes ~15-30 min):

| Metric | Value |
|---|---:|
| Correctness (LLM-judge) | 0.00 |
| Avg judge confidence | 0.78 |
| File-hit rate | 0.75 |
| Avg steps per question | 6.0 |
| Forced-final rate | 1.00 |
| Total elapsed | 146.3s (~37s/question) |

## What these numbers mean

The retrieval eval reveals a real corpus problem: BM25 surfaces docs that *describe* a symbol over the source that *defines* it. Vector and hybrid both stick at R@1 ≈ 0.31. Mixing source code and prose docs in one index is the bottleneck — not which retriever you pick. (See `weeks/05-hybrid-evals/README.md` for the full diagnosis.)

The agent eval cleanly separates two failure modes:

- **File-hit 0.75** — the agent IS finding the right files three quarters of the time. Retrieval works.
- **Correctness 0.00, Forced-final 1.00** — but Qwen 7B can't synthesize a correct answer from those files in a 5-step budget. The model loops, hits the cap, and produces hedged "I'm not sure" answers that the judge rightly marks wrong.

> **The agent can locate code but can't reason over it well enough to write a confident, correct answer at this size.** This is the headline finding. It's the difference between a 7B local model and a frontier model — the same retrieval primer, the same tools, the same prompt templates, but radically different downstream quality.

Production fixes (in increasing order of effort):

1. **Increase `max_steps`** from 5 to 10. Cuts forced-final rate; helps Correctness mildly.
2. **Use a frontier judge** (`--judge-backend anthropic`). Calibrates the headline number.
3. **Use a frontier agent** (`LANTERN_BACKEND=anthropic`). Expected to flip Correctness from 0.0 to 0.7+ on this exact harness.
4. **Swap to a real cross-encoder reranker** (`bge-reranker-v2-m3`). Lifts retrieval R@1 from 0.31 toward 0.6+.
5. **Index-time chunk-type filtering** (separate code from prose). Largest expected R@1 lift; was diagnosed as the corpus issue in week 5.

The benchmark itself doesn't ship those fixes — that's what makes it a *benchmark*. It tells you exactly what to fix next.

## Reproduce

```bash
# 0. Setup (one time)
brew install ollama
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Clone + sync deps
git clone https://github.com/jijiezzhou/ai-engineering-stack
cd ai-engineering-stack
uv sync

# 2. Build the index (~5 s)
uv run lantern index .

# 3. Retrieval benchmark (~2 s without rerank, ~12 min with rerank)
uv run lantern eval --mode retrieval --skip-rerank
uv run lantern eval --mode retrieval                    # full, slow

# 4. Agent benchmark — start small, then go all-in
uv run lantern eval --mode agent --questions 4          # ~3-6 min
uv run lantern eval --mode agent                        # all 16, ~15-30 min

# 5. Frontier comparison (requires ANTHROPIC_API_KEY)
LANTERN_BACKEND=anthropic uv run lantern eval --mode agent --questions 4
LANTERN_BACKEND=anthropic uv run lantern eval --mode agent --judge-backend anthropic

# 6. Run on YOUR codebase
cd /path/to/your-real-repo
uv run lantern index .
$EDITOR evals/your-repo.yaml          # write 5-10 questions you'd ask a new hire
uv run lantern eval --mode agent --tests evals/your-repo.yaml --repo .
```

## Methodology notes (for credibility)

- **Test set is open**: every question and reference answer in `evals/lantern.yaml` is committed in this repo. Anyone can read what's being measured and disagree.
- **Judge is the same family** by default. For a published number you'd want a stronger judge — pass `--judge-backend anthropic`. Same agent answers, more reliable grading.
- **No tuning to the test set.** Lantern's prompts and code shipped in earlier weeks; the eval was built last and the numbers are what they are.
- **`forced_final=True` is honest, not buggy**. When the agent hits max_steps and writes "I couldn't determine X", the judge marks it incorrect. That's the right behavior — incomplete answers ARE wrong from the user's perspective.
- **N=4 for the agent benchmark in this file** is for runtime tractability on a 16 GB Mac. The full 16-question run is slow but takes the same code path; reproducing it is the exercise in `weeks/08-benchmark/exercise.md`.

## What I'd run next on a real production codebase

1. **Replace `golden_answer`s with team SME-written ones.** A good benchmark is 50-100 questions, written by a domain expert.
2. **Run the agent on three model tiers** (local 7B, frontier-mini, frontier-flagship). Plot Correctness vs cost-per-query. Find the knee.
3. **Track regression on every PR.** Run `eval --mode agent --questions 8` in CI; fail the PR if Correctness drops more than X%.
4. **Layer in a cross-encoder reranker** before claiming the retrieval ceiling.

That's the agenda for "AI engineer working at a company shipping a coding agent" in 2026.

---

*This benchmark was generated by Lantern. To regenerate: `uv run lantern eval --mode agent`.*

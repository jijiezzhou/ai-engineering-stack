# Week 8 — Polish + Public Benchmark

**Goal:** stop measuring retrieval in isolation and start measuring whether the **whole agent** answers the user correctly. By the end:

```bash
uv run lantern eval --mode agent
```

…produces a published-quality benchmark — Correctness (LLM-judged), File-hit rate, Avg steps, Forced-final rate, total wall time — for whichever model you have plugged in. Compare two models, the table tells you which one to ship.

This is **the final brick** of the curriculum and the artifact that makes the repo useful to the broader community.

## Why retrieval-only evals stop being enough

Week 5's eval answers: "did the right file appear in the top-k?" That's necessary but not sufficient. A retrieval system that gives the agent the right files but the agent then writes a confused answer is a *user-facing* failure, even if Recall@5 = 1.0.

Week 8 closes the loop: run the full `agent_loop`, compare the agent's actual answer to a human-written reference, grade it.

## LLM-as-judge in one schema

```python
class _JudgeScore(BaseModel):
    correct: bool
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""
```

The judge prompt is just question + reference + agent's answer + "be strict." Same `LLM.structured()` primitive from week 2. **Reranking, scoring, judging — all the same shape.** Once you internalize this, half of "AI ops" becomes uniform.

By default the judge is the same LLM that ran the agent. For higher-quality grades pass `--judge-backend anthropic` so Claude grades Qwen — recommended for any benchmark you actually publish.

## What the agent eval reports

| Metric | What it means |
|---|---|
| **Correctness** | LLM-judge says the agent's answer matches the reference. The headline number. |
| **Avg judge confidence** | Sanity check — judge confidence under 0.6 means "even the judge isn't sure," and you should re-check the prompt. |
| **File-hit rate** | Did the agent open any of the `expected_paths` during its run? Decouples retrieval quality from answer quality. |
| **Avg steps** | How much budget the agent uses per question. Lower is cheaper. |
| **Forced-final rate** | Fraction of cases that hit max_steps without converging. High = the agent is running in circles. |
| **Total elapsed** | Wall time for the whole run. The cost-of-evals signal. |

## What ships this week

```
lantern/
├── evals.py          ← +TestCase.golden_answer, +AgentCase, +AgentEvalReport,
│                       +evaluate_agent() with LLM-as-judge
├── cli.py            ← +`eval --mode agent` flag, --max-steps, --questions,
│                       --judge-backend
└── __init__.py       ← exports unchanged

evals/
└── lantern.yaml      ← every case now has a `golden_answer` paragraph

BENCHMARKS.md         ← top-level: methodology, numbers, "where each model wins"
weeks/08-benchmark/
├── README.md         ← this file
└── exercise.md       ← run the benchmark on your own repo
```

## Try it

```bash
# Fast pass: 4 questions on local Qwen 7B (~3-6 min)
uv run lantern eval --mode agent --questions 4

# Full local pass: all 16 questions (~15-30 min)
uv run lantern eval --mode agent

# Frontier comparison: Anthropic running the agent, Anthropic also judging
LANTERN_BACKEND=anthropic uv run lantern eval --mode agent

# Mixed: Qwen runs the agent, Claude judges (most credible benchmark setup)
uv run lantern eval --mode agent --judge-backend anthropic
```

The tail of one run looks like:

```
  ✓   31s 3st  conf=0.95  Where is BM25 sparse retrieval implemented?
  ✗   58s 4st* conf=0.40  Where is reciprocal rank fusion applied?
  ✓   22s 2st  conf=0.90  How is structured output enforced from the LLM?
  ...

Agent eval — 16 questions  (ollama:qwen2.5-coder:7b)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric                              ┃  Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Correctness (LLM-judge)             │   0.62 │
│ Avg judge confidence                │   0.78 │
│ File-hit rate                       │   0.81 │
│ Avg steps per question              │    3.4 │
│ Forced-final rate                   │   0.31 │
│ Total elapsed                       │  742s  │
└─────────────────────────────────────┴────────┘
```

(Your numbers will differ. The published numbers in `BENCHMARKS.md` are the ones I ran on this Mac.)

## What the benchmark teaches

1. **Cost ≠ quality.** A 12-minute run on Qwen 7B can match a 1-minute run on Claude on simple questions but completely fall over on multi-hop ones. The full picture is per-question.
2. **LLM-as-judge isn't free, but it's worth it.** The alternative is hand-grading 16 answers each time; that's a quarter-day of human time per benchmark run.
3. **File-hit rate decouples concerns.** When Correctness is low but File-hit rate is high, the *answer-writing* stage is the bottleneck — not retrieval. Different fix.
4. **Forced-final rate is the most actionable single number.** If 30%+ of runs hit max_steps, your agent is looping. Either the model is too small, the tools are too coarse, or max_steps is too low.

## Concept → Lantern mapping

| Concept | Where Lantern uses it |
|---|---|
| LLM-as-judge | The same pattern that powers reranking (week 5) and now grading (week 8). |
| `golden_answer` field | Adds 1 paragraph per test case. Cheap to write, multiplies eval value. |
| `evaluate_agent` | Drop-in for any future agent variant — every change to agent.py can be measured against this baseline. |
| `--judge-backend` | Lets you keep the agent local while using a frontier judge. Real production benchmarks always use a stronger judge than runner. |

## Further reading

- [Anthropic — Evaluating tool use](https://docs.anthropic.com/en/docs/test-and-evaluate/strengthen-guardrails/agent-evaluations) — *production guidance on multi-step agent eval; aligns with what you just built.*
- [Eugene Yan — Evals are noisy](https://eugeneyan.com/writing/evals/) — *why your first eval set is wrong and what to do about it.*
- [BAAI — bge-reranker model card](https://huggingface.co/BAAI/bge-reranker-v2-m3) — *Qwen 7B's rerank looks weak in this benchmark; bge-reranker is the production swap.*

When you've done the [exercise](exercise.md), you're done with the curriculum. **You shipped a real, locally-runnable, MCP-deployable, benchmarked AI engineering capstone.** The next step isn't another week — it's pointing Lantern at your day-job codebase, running `lantern eval --mode agent` against questions your team actually asks, and using the gap to argue for budget.

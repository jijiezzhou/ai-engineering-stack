# Week 9 Exercise (5 min)

Feel the chunk-type filter on real queries. Compare honestly.

## 1. Re-index to populate chunk_class

```bash
uv run lantern index . --rebuild
```

The collection now stores `chunk_class` in metadata for every chunk.

## 2. Same query, three filters

The query: a symbol name. Watch the doc chunks evaporate when you add the filter.

```bash
# All kinds — docs win
uv run lantern search "_resolve_safely" --retriever bm25 -k 3

# Code only — source wins
uv run lantern search "_resolve_safely" --retriever bm25 --kinds code -k 3

# Docs only — what the docs say about it
uv run lantern search "_resolve_safely" --retriever bm25 --kinds doc -k 3
```

The first call surfaces `weeks/04-embeddings/README.md` (which mentions the symbol in prose). The second call surfaces `lantern/tools.py:52-68` (the actual function). Same retriever, same query, different *world*.

## 3. Run the retrieval eval both ways

```bash
# Default (all kinds) — what BENCHMARKS.md showed for week 5/8
uv run lantern eval --mode retrieval --skip-rerank

# Code-only — write a tiny script to filter
uv run python -c "
from lantern.evals import load_tests, evaluate
from lantern.search import hybrid_search
from pathlib import Path

cases = load_tests(Path('evals/lantern.yaml'))
report = evaluate(
    'hybrid (code-only)',
    lambda q: hybrid_search(q, repo='.', top_k=5, kinds=['code']),
    cases,
)
print(f'R@1 = {report.recall_at(1):.2f}')
print(f'R@3 = {report.recall_at(3):.2f}')
print(f'R@5 = {report.recall_at(5):.2f}')
print(f'MRR = {report.mrr:.2f}')
"
```

You should see R@5 climb to ~1.00. That's the headline number from BENCHMARKS.md — produced on your machine, in two minutes.

## 4. Watch the agent benefit

The agent's retrieval primer now defaults to `kinds=["code"]`. Run the same agent question that failed in week 8:

```bash
uv run lantern ask "Where is reciprocal rank fusion implemented?" --show-trace --max-steps 4
```

Compare its trace to a week-8 run on the same question. The first step (or the primer hits) should now point at `lantern/search.py:97-...` (the `hybrid_search` function) — not at week 5 README.md.

## 5. Disable the primer to see what changed

```bash
uv run lantern ask "Where is reciprocal rank fusion implemented?" --no-retrieval --show-trace --max-steps 4
```

Without the primer, the agent has to discover the file structure step by step. Compare the trace to step 4. **The primer is now a strict win** — same agent, different starting context.

## 6. Try the inverse — search docs only

```bash
uv run lantern search "what is RRF and why use it" --kinds doc -k 3
```

This is what the filter is *for* on the doc side too: when a user genuinely wants prose explanations, they get prose. When they want code, they get code. **The same engine, two clean modes.**

## 7. Stretch: run on YOUR repo

```bash
cd /path/to/your-real-repo
uv run lantern index . --rebuild
uv run lantern search "<some symbol you wrote>" --retriever bm25 -k 5
uv run lantern search "<same query>" --retriever bm25 --kinds code -k 5
```

If your repo mixes a `docs/` directory with source, you'll feel the same lift. **This is the change that matters most when you point Lantern at production code.**

---

When you're done, you've completed the production-retrieval extension. The benchmark numbers improved honestly (R@5 hybrid: 0.62 → 1.00). The agent's first step is now informed by source, not prose. Next extension week (week 10, if you choose) tackles the *synthesis* gap that retrieval alone can't close.

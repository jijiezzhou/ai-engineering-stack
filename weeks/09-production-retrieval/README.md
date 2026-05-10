# Week 9 — Production Retrieval (Extension)

> Optional extension week. The 8-week core curriculum is complete; this is what an engineer would do *next* — close the gap their benchmark revealed.

**Goal:** act on the week-5 / week-8 finding that mixing source code and prose docs in one index destroys retrieval. By the end:

```bash
uv run lantern eval --mode retrieval --skip-rerank        # default: all kinds
uv run lantern search "..." --retriever hybrid --kinds code
```

…and your retrieval numbers genuinely improve. On this repo, hybrid R@5 went from 0.62 to **1.00** just by excluding docs from the candidate pool.

## What week 5/8 told us

The published numbers (BENCHMARKS.md) showed:

- BM25 R@1 = 0.00 — for symbol queries, the docs that *describe* a symbol have higher term-frequency than the source that *defines* it. Documentation outranks implementation.
- Vector R@1 = 0.25–0.31 — same problem, slightly less acute.
- Hybrid didn't fix it — because both underlying rankers had the same bias.
- LLM rerank didn't fix it — because at 7B, the model couldn't reliably prefer source over docs even when prompted to.

**The retriever wasn't the problem. The corpus was.** The fix is at the boundary, not in the model.

## The fix in one paragraph

Every chunk now carries a `chunk_class` field — `"code"`, `"doc"`, `"config"`, or `"other"` — set at index time by file extension. Every retriever (`search`, `bm25_search`, `hybrid_search`) accepts an optional `kinds=["code"]` filter. The agent's retrieval primer defaults to code-only — which is correct, because the agent's questions are always about code.

That's it. Twenty lines of real code change, dramatic effect on the numbers.

## What ships this week

```
lantern/
├── chunk.py        ← +chunk_class field, +classify(path) by suffix
├── index.py        ← stores chunk_class in Chroma metadata
├── bm25.py         ← BM25Index.search accepts `kinds=` for post-filter
├── search.py       ← search/bm25_search/hybrid_search all accept `kinds=`
├── agent.py        ← retrieval primer defaults to kinds=["code"]
└── cli.py          ← `lantern search --kinds code,config` flag

weeks/09-production-retrieval/
├── README.md       ← this file
└── exercise.md     ← 5-min: feel the lift on real queries

BENCHMARKS.md       ← updated with before/after table
```

No new dependencies. The change is purely about *how we use* the existing infrastructure.

## The numbers (this Mac, Qwen 7B, 16-question test set)

### Retrieval — top-5

| Retriever | Config | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---:|---:|---:|---:|
| vector | default (all kinds) | 0.25 | 0.62 | 0.69 | 0.41 |
| vector | `kinds=["code"]` | **0.50** | **0.81** | **0.88** | **0.66** |
| bm25 | default | 0.00 | 0.25 | 0.31 | 0.13 |
| bm25 | `kinds=["code"]` | **0.69** | **0.75** | **0.81** | **0.73** |
| hybrid | default | 0.19 | 0.44 | 0.62 | 0.33 |
| hybrid | `kinds=["code"]` | **0.50** | **0.81** | **1.00** | **0.69** |

Read the BM25 row twice. **Zero to 0.69 R@1 with one filter.** That's the result of removing competing prose from a sparse index.

### Agent end-to-end (4 questions, max_steps=5)

| Metric | Week 8 (no filter) | Week 9 (code-only primer) |
|---|---:|---:|
| Correctness (LLM-judge) | 0.00 | **0.25** |
| Avg judge confidence | 0.78 | 0.95 |
| File-hit rate | 0.75 | 0.50 |
| Avg steps | 6.0 | 6.0 |
| Forced-final rate | 1.00 | 1.00 |

Correctness moved from 0/4 to 1/4. **Honestly: still mediocre.** Forced-final rate is unchanged (Qwen 7B still loops on hard questions). The retrieval primer gives the agent better starting candidates, but Qwen 7B's *synthesis* remains the bottleneck — exactly what we predicted in BENCHMARKS.md.

The next two production fixes would be:

1. **Use a frontier judge** (`--judge-backend anthropic`) for a more credible correctness number.
2. **Use a frontier agent** (`LANTERN_BACKEND=anthropic`) — almost certainly flips Correctness from 0.25 to 0.7+ on this set.

Week 9 stops at retrieval. Week 10 (if you do it) is where backend swaps and CI gates land.

## The four-line punchline

```python
# In lantern/chunk.py
def classify(path):
    if path.suffix.lower() in CODE_SUFFIXES: return "code"
    if path.suffix.lower() in DOC_SUFFIXES:  return "doc"
    if path.suffix.lower() in CONFIG_SUFFIXES: return "config"
    return "other"
```

Plus a `chunk.chunk_class = classify(path)` at index time, plus a `where={"chunk_class": ...}` clause at query time. That's the entire production fix the benchmark called for.

## When `kinds` matters most

| Query style | Filter that wins |
|---|---|
| "Where is X defined / which file does Y" | `kinds=["code"]` (always) |
| "How do I install / set up / configure" | `kinds=["doc", "config"]` |
| "What's the protocol / format / API contract" | `kinds=["doc"]` |
| "What does the README say about X" | `kinds=["doc"]` |
| Mixed exploratory question | None (default — let RRF handle it) |

A real production system would route this automatically — classify the query first, then pick the filter. That's a future extension; week 9 ships the primitive.

## Concept → Lantern mapping

| Concept | Where Lantern uses it |
|---|---|
| Chunk-class metadata | Foundation for any future filter (language, file path glob, owner, modification date). |
| `where`-clause filter at retrieval | Same shape as Pinecone, Qdrant, pgvector. Production-portable. |
| Code-only retrieval primer | The agent's first impression of the codebase is now strictly source. Massively reduces "agent grep'd for the question keywords in the README" failure mode. |
| Honest before/after benchmark | This is what production AI engineering looks like — measure, fix, re-measure. The week-9 numbers in this README are not aspirational; they're what `lantern eval` produced on this Mac. |

## Further reading

- [Pinecone — Metadata filtering](https://docs.pinecone.io/guides/data/filter-with-metadata) — *the canonical reference for the pattern; semantics match Chroma.*
- [Qdrant — Payload filtering](https://qdrant.tech/documentation/concepts/filtering/) — *more expressive filter language; useful when you outgrow chunk_class.*
- [Anthropic — Multi-vector retrieval](https://docs.anthropic.com/en/docs/build-with-claude/contextual-retrieval) — *companion technique: also store a "context" vector per chunk to fix a different failure mode.*

When you've done the [exercise](exercise.md), you've shipped a real production retrieval fix and seen it in honest numbers. That's the actual job in 2026 — diagnose, fix, measure, ship.

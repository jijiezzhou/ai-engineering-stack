# Week 5 — Hybrid Retrieval + Reranking + Evals

**Goal:** stop guessing whether retrieval is good. *Measure* it. By the end you have:

```bash
uv run lantern eval
```

…printing a table that compares vector, BM25, hybrid, and hybrid+rerank on a golden Q&A set — actual Recall@k and MRR numbers. **The first time AI-engineering quality stops being vibes and becomes a number.**

## Why three retrievers in one stack

Vector and BM25 fail on opposite cases. The right answer is to use both.

| Retriever | Wins on | Loses on |
|---|---|---|
| **Vector (cosine)** | Paraphrases. Synonyms. Conceptual queries ("how does X work?"). | Exact symbol names, error strings, file paths, abbreviations. |
| **BM25 (sparse)** | Exact tokens. Identifiers. Rare words like `_resolve_safely`. | "How does the agent decide which tool" — query has zero token overlap with the answer. |
| **Hybrid (RRF)** | Both, almost always. | Adds a few ms; that's it. |
| **+ Reranker** | Final precision — the model judges relevance directly. | Adds an LLM call per query. |

Real production systems run **hybrid → rerank**. Lantern's `lantern eval` proves it.

## Reciprocal Rank Fusion in one line

For each ranker, every doc at rank `r` contributes `1 / (k + r)` (we use `k = 60`). Sum across rankers. Highest sum wins. **Score-scale-free** — works whether your two rankers return cosines (0..1) or BM25 (unbounded) or anything else.

```python
score[doc] = sum(1 / (60 + rank_in_ranker_i) for ranker_i in [vector, bm25])
```

That's it. Three lines of Python, beats most weighted-sum schemes in practice.

## Reranking: the model is the judge

Most production RAG systems use a tiny cross-encoder model (BAAI/bge-reranker-v2-m3, ~568 MB) for reranking. Lantern uses `LLM.structured()` instead — the same primitive from week 2.

The schema:

```python
class _RerankBatch(BaseModel):
    """Relevance scores 0.0-1.0 for each passage, in order."""
    scores: list[float]
```

The prompt: question + passages + "score each from 0 to 1, be strict." One LLM call. List comes back validated.

**The lesson:** rerank is a special case of structured output. Once you have `LLM.structured()`, you have a reranker.

When to swap to a real cross-encoder:
- Latency budget < 1 s per query → cross-encoder (10-50 ms vs 5-15 s for LLM).
- High QPS → cross-encoder (cheaper per call).
- For a learning project on a 16 GB Mac → LLM-rerank wins by avoiding 700 MB of torch.

## Evals are the actual point of week 5

You can swear hybrid is better than vector. You can argue rerank earns its keep. **Don't argue — measure.** That's what an eval harness gives you.

`evals/lantern.yaml` is 16 questions about this codebase. For each:

```yaml
- question: "Which file defines the path-traversal protection?"
  expected_paths: [lantern/tools.py]
```

`evaluate(name, retriever_fn, cases)` returns `Recall@1`, `Recall@3`, `Recall@k`, `MRR`. Repeat for each retriever. Table.

This is the ONE practice that separates AI engineering from AI tinkering.

### What our eval *actually* found

Spoiler: the numbers are mediocre. Recall@1 hovers around 31% across all four retrievers. BM25 Recall@1 is **zero**. The eval revealed that on a corpus mixing source code and prose docs, **the docs win** — they have higher token frequency for the symbols people ask about.

The right response isn't to shrug. It's to *act on what the eval told you*:

- Filter by chunk type when the query is structural ("where is X defined").
- Use separate indexes for source vs prose.
- Swap to a cross-encoder reranker that's trained for the job.

None of that would be visible without the eval. **That's the lesson.**

## What ships this week

```
lantern/
├── bm25.py            ← NEW: sidecar BM25 index, persisted alongside Chroma
├── search.py          ← +bm25_search, +hybrid_search (RRF fusion)
├── rerank.py          ← NEW: rerank(query, hits) via LLM.structured()
├── evals.py           ← NEW: TestCase, EvalReport, load_tests(), evaluate()
├── index.py           ← +builds the BM25 sidecar at index time
├── cli.py             ← +`--retriever` flag on `search`; new `eval` command
└── __init__.py        ← re-exports

evals/
└── lantern.yaml       ← 16 golden questions about this repo

weeks/05-hybrid-evals/
├── README.md          ← this file
└── exercise.md        ← 5-min hands-on
```

## Try it

```bash
# Re-index to build the BM25 sidecar
uv run lantern index . --rebuild

# Run a question through each retriever
uv run lantern search "where is path traversal blocked" --retriever vector
uv run lantern search "where is path traversal blocked" --retriever bm25
uv run lantern search "where is path traversal blocked" --retriever hybrid

# The big one: full eval comparison
uv run lantern eval

# Skip the slow LLM rerank for a fast pass
uv run lantern eval --skip-rerank
```

The full `eval` takes ~1–3 minutes on a 16 GB M4 (most of it is the LLM rerank, ~10 s × 16 questions). `--skip-rerank` finishes in under 30 s.

## Concept → Lantern mapping

| Concept | Where Lantern uses it |
|---|---|
| BM25 sidecar | Week 6: agent uses `hybrid_search` to find candidate files before reading. |
| RRF fusion | Week 7: same pattern for combining MCP search results from multiple servers. |
| LLM-as-judge reranker | Week 6: same shape used for "did the agent's answer cite real code?" |
| `evaluate()` harness | Week 6 and 7: every change ships with an eval delta. No regressions allowed. |
| `expected_paths` test format | Week 8: published benchmark — Qwen 7B vs Claude vs GPT, same questions. |

## Further reading

- [Pinecone — Hybrid Search Explained](https://www.pinecone.io/learn/hybrid-search-intro/) — *clean intro to dense+sparse fusion; the diagrams are good.*
- [Cormack et al. — Reciprocal Rank Fusion (2009 paper)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — *the original; ten pages, math is gentle, the empirical case is stark.*
- [BAAI — bge-reranker model card](https://huggingface.co/BAAI/bge-reranker-v2-m3) — *what production reranking actually looks like; swap target for `lantern.rerank` in week 7.*
- [Lilian Weng — Evals are Hard](https://lilianweng.github.io/posts/2024-01-evals-are-hard/) — *why a careful eval set is more valuable than a clever model.*

When you've done the [exercise](exercise.md), you're ready for **Week 6: agents — multi-step planning, memory, guardrails.** The retrieval you just measured becomes the agent's eyes.

# Week 5 Exercise (5 min)

Watch the four retrievers fight. Then fix one.

## 1. Re-index (gives you the BM25 sidecar)

```bash
uv run lantern index . --rebuild
```

Expect: ~89 chunks, plus a new file at `~/.lantern/index/ai-engineering-stack/bm25.pkl`.

## 2. Same query, three retrievers

A query with no token overlap with the answer:

```bash
uv run lantern search "the brain that decides which tool to call" --retriever vector --top-k 3
uv run lantern search "the brain that decides which tool to call" --retriever bm25   --top-k 3
uv run lantern search "the brain that decides which tool to call" --retriever hybrid --top-k 3
```

Expected behavior:
- **vector** — finds it; `Decision` model, `LLM.call`, etc.
- **bm25** — misses badly. The query and target share zero rare tokens.
- **hybrid** — finds it (vector contributes), and ranks more confidently.

A query that's an exact symbol name:

```bash
uv run lantern search "_resolve_safely" --retriever vector --top-k 3
uv run lantern search "_resolve_safely" --retriever bm25   --top-k 3
uv run lantern search "_resolve_safely" --retriever hybrid --top-k 3
```

Expected:
- **vector** — finds it but with mediocre confidence.
- **bm25** — nails it at rank 1, score is huge (BM25 loves rare tokens).
- **hybrid** — also nails it.

> The takeaway: any single retriever has a dominant failure mode. Combining them eliminates almost all of those.

## 3. Run the full eval

```bash
uv run lantern eval
```

The actual numbers from this repo (with the default Qwen 2.5 Coder 7B rerank):

```
                    Eval — 16 questions, top-5
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━┳━━━━━━━━┓
┃ Retriever     ┃ Recall@1 ┃ Recall@3 ┃ Recall@5 ┃  MRR ┃   Time ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━━━┩
│ vector        │     0.31 │     0.69 │     0.81 │ 0.49 │   1.0s │
│ bm25          │     0.00 │     0.44 │     0.62 │ 0.22 │   0.0s │
│ hybrid        │     0.31 │     0.69 │     0.75 │ 0.48 │   0.4s │
│ hybrid+rerank │     0.31 │     0.69 │     0.81 │ 0.49 │ 756.5s │
└───────────────┴──────────┴──────────┴──────────┴──────┴────────┘
```

These are *not* the rosy numbers you'd see in a vendor demo. **They're more useful — they reveal a real problem.** Two findings:

1. **BM25 R@1 = 0.00** — for queries like "Where is `_resolve_safely` defined?", BM25 surfaces `weeks/04-embeddings/README.md` (which mentions the symbol three times in prose) above `lantern/tools.py` (which contains the actual function once, in `def _resolve_safely(...)`). Token-frequency wins over correctness.

2. **Rerank doesn't help** at this corpus + model size. R@1 stays at 0.31 even with the LLM rerank. A 7B generalist model can't reliably prefer source over docs when both are visibly relevant.

> **This is the week-5 lesson, not a failure.** Without the eval you'd have built four retrievers, demoed each one with a hand-picked query, declared victory, and shipped. The eval forced you to see that mixing source and prose in one index is the actual bottleneck — *not* which retriever you pick.

What real fixes look like:
- **Filter by chunk type at query time.** When the question is "where is X defined", search only `.py` chunks. (Future-Lantern: add a `kind` filter to retrievers.)
- **Separate indexes.** One Chroma collection for source, one for docs. Search both, fuse results — but with a stronger prior on source.
- **Production-grade reranker.** A cross-encoder (`bge-reranker-v2-m3`) trained for ranking out-performs a generalist 7B LLM by a wide margin.
- **Better tokenization for code.** Treat identifiers as a single token (don't split `_resolve_safely` into `_`, `resolve`, `safely` if you can avoid it).

These are week-7 productionization tasks. Week 5's job is just to *expose* the gap. ✓

## 4. Skip the slow rerank when iterating

```bash
uv run lantern eval --skip-rerank
```

Finishes in ~30 s. Use this while you're tuning retrieval; reserve full eval for "did this change actually help?"

## 5. Find one question your retriever fails on

Look at the eval output (or run `evaluate(...)` in Python). Pick a question where `rank=0` for vector but `rank≤3` for hybrid. Read both retrievers' top-3 hits. **Form a hypothesis** about why vector missed it. Often: query is full of high-IDF tokens that BM25 catches but the embedder doesn't weight enough.

## 6. Write your own test case

Edit `evals/lantern.yaml`, add:

```yaml
- question: "How does Lantern combine vector and BM25 ranks?"
  expected_paths: [lantern/search.py]
```

Re-run `lantern eval`. Did all four retrievers pass? Which struggled? **The eval harness is now your tool for proving every future change is an improvement.**

## 7. Stretch: evaluate against a different repo

```bash
uv run lantern index ~/some/clone
# Write your own evals/<other-repo>.yaml
uv run lantern eval --repo ~/some/clone --tests evals/<other-repo>.yaml
```

This is exactly how a real AI-eng team measures retrieval quality on production data.

---

When you're done, you have the rarest skill in AI engineering: **the ability to prove your changes work.** Week 6 is where the agent loop multiplies the value of everything you built — and where evals stop you from breaking it.

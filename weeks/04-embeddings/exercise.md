# Week 4 Exercise (5 min)

Get a feel for what semantic search gives you that `grep` can't.

## 1. Index this repo

```bash
uv run lantern index .
```

Expect: ~20 files, ~50 chunks, done in under 10 seconds. The index lives at `~/.lantern/index/ai-engineering-stack/`.

## 2. Compare grep vs semantic on the same query

The query: *"where is path traversal blocked"*. The function we want is `_resolve_safely` in `lantern/tools.py` — but it doesn't contain the words "path" or "traversal" or "blocked" together.

```bash
# grep — literal match
grep -rn "path traversal blocked" lantern/    # → nothing

# Lantern semantic search
uv run lantern search "where is path traversal blocked" -k 3
```

The top hit should be the `_resolve_safely` chunk. **Embeddings matched intent, grep matched characters.** That's the unlock.

## 3. Try queries that have NO matching keywords

```bash
uv run lantern search "the brain that decides which tool to call" -k 3
uv run lantern search "store vectors on disk" -k 3
uv run lantern search "stream tokens one at a time" -k 3
```

Expected top hits (roughly):
- `Decision` model in `agent.py`
- `_get_collection` / `INDEX_ROOT` in `index.py`
- `LLM.stream()` in `llm.py`

The model never wrote those exact phrases. Embeddings still find them.

## 4. Watch chunking work

```bash
uv run lantern search "what is a Lantern tool" -k 5
```

Look at the `kind:` of each hit. You should see `class:ToolSpec`, `class:read_file`, etc. — each one is a coherent semantic unit, not a sliced-up window. **Tree-sitter chunking is doing real work.**

## 5. Test where it fails

```bash
uv run lantern search "what's the weather in tokyo" -k 3
```

You'll still get hits — vector search always returns the top-k, even if the query is irrelevant. Note the `score=` values. Hits below ~0.5 are usually noise.

> Vector search has no "no match" answer. Production systems either threshold scores or use a reranker (week 5). For now, just notice the limit.

## 6. Index a different repo

```bash
uv run lantern index ~/some/cloned/repo
uv run lantern search "where is the entry point" -r ~/some/cloned/repo -k 5
```

Each repo gets its own collection at `~/.lantern/index/<repo-name>/`. Switching repos means switching `--repo`.

## 7. Stretch: re-index after a code change

Edit any file (add a function, rename a class), then:

```bash
uv run lantern index .
uv run lantern search "<query that should hit your edit>" -k 3
```

Notice that `index` was idempotent — same chunk IDs, upserted in place. Use `--rebuild` only if the schema changes.

---

When you're done, you've internalized RAG's first half (retrieval). Week 5 builds on this with hybrid search (BM25 + vectors), a reranker, and an **eval harness** that quantifies "is this retrieval actually working?"

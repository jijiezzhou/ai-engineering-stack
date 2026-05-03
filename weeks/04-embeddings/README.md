# Week 4 — Embeddings + Code-Aware Chunking

**Goal:** stop relying on substring grep. Make Lantern find the right code by *meaning*. By the end:

```bash
uv run lantern index .
uv run lantern search "where is path traversal blocked" -k 3
```

…and the top hit is `_resolve_safely` in `lantern/tools.py` — even though the words "path traversal" appear nowhere near that function.

## Why this is the leap from "grep" to "RAG"

Grep matches characters. Embeddings match *meaning*.

| Query | Grep finds | Embedding finds |
|---|---|---|
| `_resolve_safely` | the function (literal name match) | same thing |
| `"where is path traversal blocked"` | nothing | `_resolve_safely` |
| `"how does the model pick a tool"` | random `tool` mentions | the `Decision` schema in `agent.py` |
| `"streaming an LLM response"` | nothing useful | `LLM.stream()` in `llm.py` |

Once you have semantic search, every later AI-engineering pattern (RAG, agents that explore code, deep-research over docs) becomes a special case of "embed and search."

## Vector embeddings, in one paragraph

A model maps every chunk of text to a vector — a list of ~768 floats. Texts about similar things land near each other in that 768-dimensional space. **Semantic similarity becomes geometric distance.** "How do I make a list?" sits near "Python `list()` constructor"; far from "how to ship a docker image."

Two operations are all you need:

1. **Embed text → vector.** Done by an embedding model. We use `nomic-embed-text` via Ollama (local, free, 768-dim, runs on your M4 in milliseconds).
2. **Find the nearest vectors.** Done by a vector database. We use Chroma (embedded, persists to disk, no server).

## Why chunking matters more than the model

You do **not** embed whole files. Files are too big and mix concerns. You chunk them — and the chunk boundaries decide whether retrieval is good or garbage.

| Strategy | Quality |
|---|---|
| 100-line fixed-size windows | Bad. Chunks slice through functions; embeddings drift. |
| 1500-char fixed-size with newline breaks | OK for prose / configs. |
| **Tree-sitter: one chunk per function/class** | **Best.** Each chunk is semantically coherent — a function with its docstring and body, a class with its methods. |

Lantern uses tree-sitter for `.py` files (its own language) and falls back to fixed-size for everything else. The first chunk of every Python file is a "header" containing imports + module-level code — equally important to embed.

## What ships this week

```
lantern/
├── chunk.py           ← NEW: Chunk dataclass + chunk_file() (tree-sitter / fixed)
├── index.py           ← NEW: index_repo() — walks files, embeds, persists to Chroma
├── search.py          ← NEW: Hit dataclass + search(query, repo)
├── cli.py             ← +`lantern index` and `lantern search` subcommands
└── __init__.py        ← re-exports

weeks/04-embeddings/
├── README.md          ← this file
└── exercise.md        ← 5-min hands-on
```

Storage lives at `~/.lantern/index/<repo-name>/`. One global cache, doesn't pollute target repos.

## The two new commands

```bash
# Index any repo (idempotent — safe to re-run after edits)
uv run lantern index .
uv run lantern index ~/some/clone --rebuild   # wipe and start fresh

# Search it
uv run lantern search "<natural language query>" -r .  -k 5
```

`search` returns ranked hits with `path:start-end`, the kind (`function | class | header | fixed`), the symbol name when known, the cosine score, and a 5-line preview.

## Why we don't (yet) wire `search` into `ask`

You could imagine `ask` calling `search` to find the right file before reading it. That's exactly week 5's job — wrap retrieval, reranking, and an eval harness around it. For week 4, we keep `search` standalone so you can:

1. Watch retrieval work (or fail) in isolation.
2. Compare semantic search to grep on the same query.
3. Build intuition for how chunking and queries interact before adding more layers.

## Concept → Lantern mapping

| Concept | Where Lantern uses it |
|---|---|
| `Chunk` dataclass | Week 5: every retrieval/eval row is anchored to one. |
| `index_repo()` | Week 5: same call, plus a BM25 sidecar index. |
| `search()` | Week 5: replaced by `hybrid_search()` that combines vectors + BM25 + rerank. |
| Cosine score | Week 5: input feature for the reranker; week 7: cost/quality trade-off knob. |
| Tree-sitter chunks | Week 6: the agent reads a single chunk before opening the whole file. |

## Cost & latency on a 16 GB M4

| Operation | Speed |
|---|---|
| One embedding (nomic-embed-text) | ~3-15 ms |
| Index this repo (~20 files, ~50 chunks) | ~5-10 s first time |
| One `search` query | ~30-100 ms (most is the embedding call) |
| Disk used per repo | ~5-50 MB |

## Further reading

- [Nomic — nomic-embed-text-v1 paper](https://arxiv.org/abs/2402.01613) — *the model. Open weights, beats `ada-002` on MTEB. Worth knowing the people behind your default embedder.*
- [Chroma docs — collections](https://docs.trychroma.com/usage-guide#using-collections) — *what `get_or_create_collection` actually does, including HNSW vs IVF.*
- [Greg Kamradt — 5 Levels of Text Splitting](https://www.youtube.com/watch?v=8OJC21T2SL4) — *one-hour video; the cleanest explanation of chunking strategies and where each fails.*

When you've done the [exercise](exercise.md), you're ready for **Week 5: hybrid retrieval + reranking + evals.**

"""
Week 4–5 — search over an indexed repository.

Three retrievers, all return `list[Hit]`:

    search(query, repo)         — dense vector (cosine over nomic-embed-text)
    bm25_search(query, repo)    — sparse lexical (BM25 over tokens)
    hybrid_search(query, repo)  — Reciprocal Rank Fusion of the two

Plus `lantern.rerank.rerank()` for an LLM-judged re-ranking pass on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lantern.bm25 import BM25Index, bm25_path
from lantern.index import EMBED_MODEL, INDEX_ROOT, _get_collection


@dataclass
class Hit:
    path: str
    start_line: int
    end_line: int
    kind: str
    name: str
    content: str
    score: float  # interpretation depends on retriever; higher = better


# ---------------------------------------------------------------- vector

def search(
    query: str,
    *,
    repo: Path | str = ".",
    top_k: int = 5,
    embed_model: str = EMBED_MODEL,
) -> list[Hit]:
    """Top-k cosine hits for `query` in the dense vector index for `repo`."""
    repo_path = Path(repo).resolve()
    coll = _get_collection(repo_path, reset=False)

    from ollama import Client
    client = Client()
    q_emb = client.embed(model=embed_model, input=query)["embeddings"][0]

    res = coll.query(query_embeddings=[q_emb], n_results=top_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]

    return [
        Hit(
            path=str(meta.get("path", "")),
            start_line=int(meta.get("start_line", 0)),
            end_line=int(meta.get("end_line", 0)),
            kind=str(meta.get("kind", "")),
            name=str(meta.get("name", "") or ""),
            content=doc,
            score=1.0 - float(dist),
        )
        for doc, meta, dist in zip(docs, metas, distances)
    ]


# ------------------------------------------------------------------ bm25

def bm25_search(query: str, *, repo: Path | str = ".", top_k: int = 5) -> list[Hit]:
    """Top-k BM25 hits — sparse lexical retrieval. Catches exact symbol names."""
    repo_path = Path(repo).resolve()
    path = bm25_path(INDEX_ROOT, repo_path.name)
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 index missing at {path}. Run `lantern index <repo>` to build it."
        )
    bm25 = BM25Index.load(path)
    raw = bm25.search(query, top_k=top_k)
    return [
        Hit(
            path=c.path,
            start_line=c.start_line,
            end_line=c.end_line,
            kind=c.kind,
            name=c.name,
            content=c.content,
            score=float(score),
        )
        for c, score in raw
    ]


# ---------------------------------------------------------------- hybrid

def hybrid_search(
    query: str,
    *,
    repo: Path | str = ".",
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[Hit]:
    """Reciprocal Rank Fusion of the vector and BM25 rankings.

    For each ranker, every document at rank `r` contributes `1 / (rrf_k + r)`
    to its score. Document ordering then comes from summing these contributions
    across rankers. Robust to score-scale differences between rankers
    (cosine 0–1 vs BM25 unbounded), unlike weighted-sum.
    """
    pool = max(top_k * 3, 15)
    vec = search(query, repo=repo, top_k=pool)
    try:
        sparse = bm25_search(query, repo=repo, top_k=pool)
    except FileNotFoundError:
        sparse = []  # graceful: hybrid degrades to vector-only

    scores: dict[str, float] = {}
    by_key: dict[str, Hit] = {}

    def key(h: Hit) -> str:
        return f"{h.path}:{h.start_line}-{h.end_line}"

    for rank, h in enumerate(vec):
        k = key(h)
        scores[k] = scores.get(k, 0.0) + 1.0 / (rrf_k + rank)
        by_key[k] = h
    for rank, h in enumerate(sparse):
        k = key(h)
        scores[k] = scores.get(k, 0.0) + 1.0 / (rrf_k + rank)
        by_key.setdefault(k, h)

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    return [
        Hit(
            path=by_key[k].path,
            start_line=by_key[k].start_line,
            end_line=by_key[k].end_line,
            kind=by_key[k].kind,
            name=by_key[k].name,
            content=by_key[k].content,
            score=score,
        )
        for k, score in ranked
    ]

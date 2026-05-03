"""
Week 4 — semantic search over an indexed repository.

`search("where is path traversal blocked")` finds `_resolve_safely` even
though the words "path traversal" never appear in that function. That's
the unlock embeddings give you over `grep`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lantern.index import EMBED_MODEL, _get_collection


@dataclass
class Hit:
    path: str
    start_line: int
    end_line: int
    kind: str
    name: str
    content: str
    score: float  # 1.0 = identical direction in embedding space, 0.0 = orthogonal


def search(
    query: str,
    *,
    repo: Path | str = ".",
    top_k: int = 5,
    embed_model: str = EMBED_MODEL,
) -> list[Hit]:
    """Top-k semantic hits for `query` in the index for `repo`."""
    repo_path = Path(repo).resolve()
    coll = _get_collection(repo_path, reset=False)

    from ollama import Client
    client = Client()
    q_emb = client.embed(model=embed_model, input=query)["embeddings"][0]

    res = coll.query(query_embeddings=[q_emb], n_results=top_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]

    hits: list[Hit] = []
    for doc, meta, dist in zip(docs, metas, distances):
        hits.append(Hit(
            path=str(meta.get("path", "")),
            start_line=int(meta.get("start_line", 0)),
            end_line=int(meta.get("end_line", 0)),
            kind=str(meta.get("kind", "")),
            name=str(meta.get("name", "") or ""),
            content=doc,
            score=1.0 - float(dist),
        ))
    return hits

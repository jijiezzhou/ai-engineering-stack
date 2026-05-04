"""
Week 4 — embed and persist code chunks for semantic search.

Storage:  ~/.lantern/index/<repo-name>/   (Chroma persistent client)
Embedder: nomic-embed-text via the local Ollama daemon (768-dim, cosine).

The index is per-repo and idempotent: re-running `index_repo` upserts
chunks, so editing files and re-indexing gives you fresh vectors without
wiping the whole collection. Use `rebuild=True` to wipe.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Iterable, Optional

from lantern.chunk import Chunk, chunk_file
from lantern.tools import SKIP_DIRS, SKIP_SUFFIXES

INDEX_ROOT = Path.home() / ".lantern" / "index"
EMBED_MODEL = "nomic-embed-text"
EMBED_BATCH = 32

# Suffixes worth indexing. Source code, docs, configs. Skip binaries / lockfiles.
INDEX_SUFFIXES = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt",
    ".rb", ".sh", ".sql", ".md", ".toml", ".yaml", ".yml", ".json",
    ".html", ".css", ".cfg", ".ini",
})


def index_repo(
    repo: Path,
    *,
    embed_model: str = EMBED_MODEL,
    rebuild: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Walk `repo`, chunk every source file, embed each chunk, persist to Chroma.

    Returns a small stats dict. Pass a `progress(done, total)` callback for
    a per-batch heartbeat."""
    repo = repo.resolve()
    coll = _get_collection(repo, reset=rebuild)

    files = list(_walk_indexable(repo))
    chunks: list[Chunk] = []
    for f in files:
        chunks.extend(chunk_file(f))
    if not chunks:
        return {"files": len(files), "chunks": 0}

    from ollama import Client
    client = Client()

    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i:i + EMBED_BATCH]
        # `embed` is the modern API (`embeddings` is deprecated) and accepts a
        # list — one HTTP call per batch instead of N.
        resp = client.embed(model=embed_model, input=[c.content for c in batch])
        embeddings = resp["embeddings"]
        coll.upsert(
            ids=[_chunk_id(c) for c in batch],
            embeddings=embeddings,
            documents=[c.content for c in batch],
            metadatas=[
                {
                    "path": c.path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "kind": c.kind,
                    "name": c.name,
                }
                for c in batch
            ],
        )
        if progress:
            progress(min(i + EMBED_BATCH, len(chunks)), len(chunks))

    # Build the BM25 sidecar. Cheap (sub-second on this repo) and dependent on
    # the full corpus, so we rebuild every time rather than try to incrementalize.
    from lantern.bm25 import BM25Index, bm25_path
    BM25Index(chunks).save(bm25_path(INDEX_ROOT, repo.name))

    return {"files": len(files), "chunks": len(chunks)}


def _walk_indexable(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        if path.suffix not in INDEX_SUFFIXES:
            continue
        yield path


def _chunk_id(c: Chunk) -> str:
    h = hashlib.sha1()
    h.update(c.path.encode("utf-8"))
    h.update(f"{c.start_line}-{c.end_line}-{c.kind}-{c.name}".encode("utf-8"))
    return h.hexdigest()[:16]


def _get_collection(repo: Path, *, reset: bool):
    import chromadb
    from chromadb.config import Settings

    db_path = INDEX_ROOT / repo.name
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    if reset:
        try:
            client.delete_collection("code")
        except Exception:  # noqa: BLE001 — chroma raises a generic NotFound here
            pass
    return client.get_or_create_collection(name="code", metadata={"hnsw:space": "cosine"})

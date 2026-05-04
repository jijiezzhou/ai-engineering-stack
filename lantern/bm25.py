"""
Week 5 — sidecar BM25 index.

BM25 is the gold standard for sparse, lexical retrieval — token overlap with
IDF weighting and document-length normalization. It catches things vector
search misses: exact symbol names, abbreviations, file paths, error strings.

Stored alongside Chroma at `~/.lantern/index/<repo>/bm25.pkl`. We rebuild it
fully each `index_repo` call (BM25's IDF weights depend on the whole corpus,
so partial updates aren't trivial — and on a 16 GB Mac, building over even
~1k chunks is sub-second).
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from lantern.chunk import Chunk

# Tokenizer: lowercase, then split on word boundaries. Keeps `_resolve_safely`
# whole; splits camelCase the boring way (snake_case wins, code is mostly
# Python in this repo).
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def bm25_path(index_root: Path, repo_name: str) -> Path:
    return index_root / repo_name / "bm25.pkl"


class BM25Index:
    """In-memory BM25Okapi over chunks, with persist/load helpers."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.tokenized = [tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "tokenized": self.tokenized}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls.__new__(cls)
        instance.chunks = data["chunks"]
        instance.tokenized = data["tokenized"]
        instance.bm25 = BM25Okapi(instance.tokenized)
        return instance

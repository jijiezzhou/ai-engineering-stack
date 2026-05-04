"""
Week 5 — LLM-based reranking.

Production-grade reranking uses a cross-encoder model (e.g. BAAI/bge-reranker-
v2-m3) — small, fast, designed for it. We use `LLM.structured()` instead so
the curriculum stays:
  - dependency-light (no torch / sentence-transformers)
  - backend-uniform (works on Qwen 7B and on Anthropic with one env-var swap)
  - pedagogically clear: rerank IS structured output. The model returns one
    relevance score per passage; structured output keeps that strict.

For a real product, swap to `FlagEmbedding`'s BGE reranker — ~10× faster and a
notch better on standard benchmarks. The interface here (`rerank(query, hits)`)
stays the same.

One LLM call per query (passages are batched into a single prompt). On Qwen 7B
a rerank over the top-10 of a hybrid hit list takes ~10 s. Worth it for the
quality jump (see `lantern eval`).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from lantern.llm import LLM
from lantern.search import Hit


class _RerankBatch(BaseModel):
    """Relevance scores 0.0-1.0 for each passage, in the order given."""

    scores: list[float] = Field(
        description=(
            "One score in [0.0, 1.0] per passage, in the same order. "
            "Same length as the input list. 1.0 = directly answers the question. "
            "0.5 = tangentially related. 0.0 = irrelevant."
        ),
    )


_SYSTEM = (
    "You score how relevant code passages are to a question. Be strict: only "
    "the 1-3 passages out of many that *directly* answer the question should "
    "score above 0.7. Most passages should score below 0.3.\n\n"
    "When a question asks 'where is X defined', 'which file does Y', or "
    "'how does Z work', the SOURCE FILE that implements X/Y/Z (a `.py` chunk "
    "of kind=function/class/header) scores highest. Documentation files "
    "(`.md`, README, exercise) that merely *describe* or *mention* the symbol "
    "should score around 0.3 — they describe but don't implement. The actual "
    "implementation always wins over prose about it."
)


def rerank(
    query: str,
    hits: list[Hit],
    *,
    llm: Optional[LLM] = None,
    top_k: Optional[int] = None,
    max_passage_chars: int = 1200,
) -> list[Hit]:
    """Re-score `hits` by LLM-judged relevance. Returns the list reordered.

    If `top_k` is given, returns at most that many. Original `hits` are not
    mutated."""
    if not hits:
        return []
    llm = llm or LLM()

    passages_text = "\n\n".join(
        f"[{i}] `{h.path}:{h.start_line}-{h.end_line}` ({h.kind}{':' + h.name if h.name else ''})\n"
        f"{h.content[:max_passage_chars]}"
        for i, h in enumerate(hits)
    )
    prompt = (
        f"Question: {query}\n\n"
        f"Passages:\n\n{passages_text}\n\n"
        f"Score the relevance of each passage to the question. Return exactly "
        f"{len(hits)} scores, one per passage, in order."
    )

    try:
        result = llm.structured(prompt, _RerankBatch, system=_SYSTEM, temperature=0.0)
        scores = list(result.scores)
        if len(scores) < len(hits):
            scores = scores + [0.0] * (len(hits) - len(scores))
        elif len(scores) > len(hits):
            scores = scores[:len(hits)]
    except Exception:  # noqa: BLE001 — fall back to upstream score
        scores = [h.score for h in hits]

    rescored = [
        Hit(
            path=h.path,
            start_line=h.start_line,
            end_line=h.end_line,
            kind=h.kind,
            name=h.name,
            content=h.content,
            score=float(new_score),
        )
        for h, new_score in zip(hits, scores)
    ]
    rescored.sort(key=lambda h: -h.score)
    return rescored[:top_k] if top_k else rescored

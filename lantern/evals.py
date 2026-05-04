"""
Week 5 — eval harness.

Test set format (YAML):

    - question: "Which file defines path-traversal protection?"
      expected_paths: [lantern/tools.py]

For each question, a retriever returns a ranked list of `Hit`s. A test passes
at rank `r` if any hit at position `r` (1-indexed) has a path that ends with
one of the question's `expected_paths`.

Metrics:
    Recall@k — fraction of questions where an expected hit appeared in top-k.
    MRR      — mean reciprocal rank of the FIRST expected hit (0 if missed).

Both are classic. No external eval framework needed. Add `braintrust` /
`phoenix` integration in week 7 if you want dashboards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from lantern.search import Hit


@dataclass
class TestCase:
    question: str
    expected_paths: list[str]


@dataclass
class CaseResult:
    question: str
    expected_paths: list[str]
    hits: list[Hit]
    rank: int  # 1-indexed rank of first expected hit, or 0 if not found in top-k

    def hit_at(self, k: int) -> bool:
        return 0 < self.rank <= k


@dataclass
class EvalReport:
    name: str
    cases: list[CaseResult]

    def recall_at(self, k: int) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.hit_at(k)) / len(self.cases)

    @property
    def mrr(self) -> float:
        if not self.cases:
            return 0.0
        return sum((1.0 / c.rank) if c.rank > 0 else 0.0 for c in self.cases) / len(self.cases)


def load_tests(path: Path) -> list[TestCase]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    return [
        TestCase(question=item["question"], expected_paths=list(item["expected_paths"]))
        for item in raw
    ]


def _matches(hit: Hit, expected: list[str]) -> bool:
    """A hit matches if its path ends with any expected path. Lets us write
    test sets in repo-relative paths even when the index stores absolutes."""
    p = hit.path.replace("\\", "/")
    return any(p.endswith(e.replace("\\", "/")) for e in expected)


def evaluate(
    name: str,
    retrieve_fn: Callable[[str], list[Hit]],
    tests: list[TestCase],
) -> EvalReport:
    """Run `retrieve_fn` against every test; return a report."""
    cases: list[CaseResult] = []
    for tc in tests:
        hits = retrieve_fn(tc.question)
        rank = 0
        for i, h in enumerate(hits, 1):
            if _matches(h, tc.expected_paths):
                rank = i
                break
        cases.append(CaseResult(
            question=tc.question,
            expected_paths=tc.expected_paths,
            hits=hits,
            rank=rank,
        ))
    return EvalReport(name=name, cases=cases)

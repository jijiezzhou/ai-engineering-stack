"""
Week 5 — retrieval eval harness. Week 8 — agent eval with LLM-as-judge.

Test set format (YAML):

    - question: "Which file defines path-traversal protection?"
      expected_paths: [lantern/tools.py]
      golden_answer: "Implemented by `_resolve_safely` in lantern/tools.py..."

For each question, two evals are possible:

  1. **Retrieval eval** (week 5): a retriever returns ranked Hits; we score
     by Recall@k and MRR using `expected_paths` as ground truth.

  2. **Agent eval** (week 8): the full `agent_loop` runs end-to-end; the
     final answer is graded against `golden_answer` by LLM-as-judge.
     Reports correctness, average steps, forced-final rate, and the
     fraction of expected files the agent actually opened.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml
from pydantic import BaseModel, Field

from lantern.search import Hit


# Match repo-relative source-file paths that appear in tool output / answers.
# Conservative: requires a real-looking suffix. `lantern/tools.py:52` matches
# `lantern/tools.py`. Catches what `_collect_seen_paths()` cares about.
_PATH_RE = re.compile(
    r"\b[A-Za-z0-9_./\-]+\.(?:py|md|yaml|yml|toml|json|txt|sh|js|jsx|ts|tsx|go|rs|java|kt|rb|sql|html|css|cfg|ini)\b"
)


# ---------------------------------------------------------------- shared types

@dataclass
class TestCase:
    question: str
    expected_paths: list[str]
    golden_answer: str = ""  # optional — only used by agent eval


def load_tests(path: Path) -> list[TestCase]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    return [
        TestCase(
            question=item["question"],
            expected_paths=list(item.get("expected_paths", [])),
            golden_answer=item.get("golden_answer", ""),
        )
        for item in raw
    ]


# ---------------------------------------------------------------- retrieval eval (week 5)

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


def _matches(hit: Hit, expected: list[str]) -> bool:
    p = hit.path.replace("\\", "/")
    return any(p.endswith(e.replace("\\", "/")) for e in expected)


def evaluate(
    name: str,
    retrieve_fn: Callable[[str], list[Hit]],
    tests: list[TestCase],
) -> EvalReport:
    """Run `retrieve_fn` against every test; return Recall@k / MRR report."""
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


# ---------------------------------------------------------------- agent eval (week 8)

class _JudgeScore(BaseModel):
    """LLM-as-judge verdict on whether an agent's answer matches the reference."""

    correct: bool = Field(
        description=(
            "True only if the agent's answer matches the reference on the key "
            "facts (file names, symbol names, what the code actually does). "
            "Hallucinated file paths are NEVER correct. Saying 'not implemented' "
            "when the reference says it IS implemented is NEVER correct."
        ),
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Your confidence in this verdict, 0.0 to 1.0.",
    )
    notes: str = Field(
        default="",
        description="One short sentence on what was right or wrong. Empty if the answer was clean.",
    )


JUDGE_SYSTEM = (
    "You are a strict but fair grader for AI-engineering coding agent answers. "
    "You compare the agent's answer to a reference (the ground truth). "
    "Match on substance, not wording: if the agent named the right file and "
    "function but used different prose, that's still correct."
)


JUDGE_TEMPLATE = (
    "Question: {question}\n\n"
    "Reference answer (ground truth):\n{golden_answer}\n\n"
    "Agent's answer:\n{agent_answer}\n\n"
    "Grade the agent's answer."
)


@dataclass
class AgentCase:
    question: str
    golden_answer: str
    expected_paths: list[str]
    agent_answer: str
    correct: bool
    confidence: float
    judge_notes: str
    n_steps: int
    forced_final: bool
    elapsed_s: float
    paths_seen: list[str]  # path args + paths surfaced in tool outputs + paths in final answer

    @property
    def hit_expected_file(self) -> bool:
        """True if any expected path appeared in the agent's work — either as
        an explicit tool argument, in tool output, or in the final answer.
        Decouples 'did the agent look at the right files' from 'did the
        answer mention them in prose'."""
        if not self.expected_paths:
            return True
        blob = " ".join(self.paths_seen)
        for ep in self.expected_paths:
            ep_norm = ep.replace("\\", "/")
            if any(p.replace("\\", "/").endswith(ep_norm) or ep_norm in p.replace("\\", "/")
                   for p in self.paths_seen):
                return True
            if ep_norm in blob:
                return True
        return False


@dataclass
class AgentEvalReport:
    name: str  # e.g. "ollama:qwen2.5-coder:7b"
    cases: list[AgentCase] = field(default_factory=list)

    @property
    def correctness(self) -> float:
        return sum(1 for c in self.cases if c.correct) / len(self.cases) if self.cases else 0.0

    @property
    def avg_confidence(self) -> float:
        return sum(c.confidence for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def avg_steps(self) -> float:
        return sum(c.n_steps for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def forced_rate(self) -> float:
        return sum(1 for c in self.cases if c.forced_final) / len(self.cases) if self.cases else 0.0

    @property
    def file_hit_rate(self) -> float:
        """Fraction of cases where the agent opened at least one expected file."""
        return sum(1 for c in self.cases if c.hit_expected_file) / len(self.cases) if self.cases else 0.0

    @property
    def total_elapsed_s(self) -> float:
        return sum(c.elapsed_s for c in self.cases)


def evaluate_agent(
    cases: list[TestCase],
    *,
    repo: str | Path = ".",
    llm=None,
    judge=None,
    max_steps: int = 5,
    on_case: Optional[Callable[[AgentCase], None]] = None,
) -> AgentEvalReport:
    """Run the multi-step agent on each case and grade the answer.

    `judge` defaults to the same LLM that's running the agent. For more
    reliable grading, pass a stronger model (e.g. an Anthropic-backed LLM).
    `on_case(case)` is called after each grading; useful for live progress."""
    from lantern.agent import agent_loop
    from lantern.llm import LLM

    repo_path = Path(repo).resolve()
    llm = llm or LLM()
    judge = judge or llm

    report = AgentEvalReport(name=f"{llm.backend}:{llm.model}")
    for tc in cases:
        if not tc.golden_answer:
            continue
        started = time.perf_counter()
        result = agent_loop(tc.question, repo=repo_path, llm=llm, max_steps=max_steps)
        elapsed = time.perf_counter() - started

        paths_seen: list[str] = []
        for step in result.steps:
            if step.decision.path:
                paths_seen.append(step.decision.path)
            if step.tool_output:
                paths_seen.extend(_PATH_RE.findall(step.tool_output))
        paths_seen.extend(_PATH_RE.findall(result.answer))

        try:
            score = judge.structured(
                JUDGE_TEMPLATE.format(
                    question=tc.question,
                    golden_answer=tc.golden_answer,
                    agent_answer=result.answer,
                ),
                _JudgeScore,
                system=JUDGE_SYSTEM,
                temperature=0.0,
            )
            correct = score.correct
            confidence = score.confidence
            notes = score.notes
        except Exception as e:  # noqa: BLE001
            correct = False
            confidence = 0.0
            notes = f"judge failed: {e}"

        case = AgentCase(
            question=tc.question,
            golden_answer=tc.golden_answer,
            expected_paths=tc.expected_paths,
            agent_answer=result.answer,
            correct=correct,
            confidence=confidence,
            judge_notes=notes,
            n_steps=len(result.steps),
            forced_final=result.forced_final,
            elapsed_s=elapsed,
            paths_seen=paths_seen,
        )
        report.cases.append(case)
        if on_case:
            on_case(case)
    return report

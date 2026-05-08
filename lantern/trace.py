"""
Week 7 — JSONL trace for agent runs.

Every `agent_loop` invocation can take a `trace: Trace | None`. If given, the
loop writes one JSON line per event to `~/.lantern/traces/<run_id>.jsonl`:

    {"run": "...", "step": 0, "ts": 0.124, "kind": "decision", ...}
    {"run": "...", "step": 0, "ts": 4.521, "kind": "tool_output", ...}
    {"run": "...", "step": 1, "ts": 4.522, "kind": "decision", ...}
    {"run": "...", "step": 1, "ts": 9.117, "kind": "answer", ...}

Why JSONL: append-only, line-oriented, greppable, plays nicely with `jq`,
trivial to consume programmatically. No bespoke trace viewer needed —
`lantern trace <id>` is enough; for richer dashboards, pipe to braintrust /
Phoenix later.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

TRACE_ROOT = Path.home() / ".lantern" / "traces"


def new_run_id() -> str:
    """Sortable, human-readable, collision-resistant id."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


@dataclass
class Trace:
    """An append-only JSONL trace for a single agent_loop run."""

    run_id: str = field(default_factory=new_run_id)
    root: Path = field(default_factory=lambda: TRACE_ROOT)
    started_at: float = field(default_factory=time.time)

    @property
    def path(self) -> Path:
        return self.root / f"{self.run_id}.jsonl"

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, step: int, kind: str, **data: Any) -> None:
        """Append one event. `kind` is free-form ('decision', 'tool_output',
        'answer', 'forced_final', 'error', etc.)."""
        line = {
            "run": self.run_id,
            "step": step,
            "ts": round(time.time() - self.started_at, 4),
            "kind": kind,
            **data,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")


def list_runs(root: Path = TRACE_ROOT, limit: int = 20) -> list[Path]:
    """Most-recent-first list of trace files."""
    if not root.exists():
        return []
    files = sorted(root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def read_run(run_id_or_path: str | Path) -> list[dict]:
    """Load all events for a run. Accepts either a bare run_id or a path."""
    p = Path(run_id_or_path)
    if not p.exists():
        p = TRACE_ROOT / f"{run_id_or_path}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"trace not found: {run_id_or_path}")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_runs(root: Path = TRACE_ROOT) -> Iterable[tuple[Path, dict]]:
    """Yield (path, summary_dict) for each trace, oldest first. Summary
    includes question, final_kind ('answer' | 'forced_final'), num_steps."""
    for path in sorted(root.glob("*.jsonl") if root.exists() else []):
        try:
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception:  # noqa: BLE001
            continue
        question: Optional[str] = None
        final_kind: Optional[str] = None
        max_step = -1
        for ev in events:
            max_step = max(max_step, int(ev.get("step", -1)))
            if ev.get("kind") == "run_start":
                question = ev.get("question")
            elif ev.get("kind") in ("answer", "forced_final"):
                final_kind = ev["kind"]
        yield path, {
            "run_id": path.stem,
            "question": question,
            "final_kind": final_kind,
            "num_steps": max_step + 1,
        }

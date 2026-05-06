"""
Week 6 — multi-step onboarding agent.

The week-3 `ask()` was a fixed two-shot: model picks one tool, we run it,
model writes the answer. That's good for "where is X defined" but breaks on
"trace this call chain across three files" — the agent can't follow up.

Week 6 wraps a *loop* around the same primitive:

    while not done and step < max_steps:
        decision = LLM.structured(prompt + history, Decision)
        if decision.next_action == "final_answer":
            return decision.answer
        output = run_tool(decision)
        history.append(Step(decision, output))

Three pieces make this work in practice:

1. **Reasoning before action.** Decision now has a `reasoning` field — the
   model writes one short sentence about what it's trying to learn before
   it picks a tool. Forces a tiny chain-of-thought; cheap, and the trace is
   readable for debugging.

2. **Retrieval primer.** Before step 0, we run `hybrid_search` for the user's
   question and put the top hits' file paths and snippets in the prompt.
   The model now starts with informed candidates instead of guessing
   `read_file('main.py')` on the first turn.

3. **Step-aware memory.** Each iteration sees a numbered list of prior
   steps with their reasoning, action, and (truncated) tool output. The
   prompt-builder caps total context to keep us safely under 32K tokens.

The week-3 single-step `ask()` is preserved — `lantern ask --single-step`
keeps that behavior. The default is now `agent_loop()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from lantern.llm import LLM
from lantern.tools import grep, list_dir, read_file

AGENT_SYSTEM = (
    "You are a senior engineer onboarding to an unfamiliar codebase. "
    "You answer questions by inspecting code with three tools: read_file "
    "(open a file), list_dir (list a directory), grep (find a substring "
    "across files). Each step you take should genuinely move you closer to "
    "answering — no busywork. When you have enough information, set "
    "next_action='final_answer' and write a concrete, sourced answer. "
    "Be specific: name real files, real symbols, real line numbers."
)

# Per-tool-output cap when rendering history. Keeps prompts under 32K even
# after several steps. Tunable; see _render_history.
HISTORY_OUTPUT_CHARS = 2000

# How many top retrieval hits to put in the primer. 5 is enough to bias the
# first step without dominating the prompt.
PRIMER_TOP_K = 5


class Decision(BaseModel):
    """One step of the agent loop: think, then act."""

    reasoning: str = Field(
        description=(
            "One short sentence about what you're trying to learn THIS step. "
            "Don't restate the user's question; describe your immediate sub-goal."
        ),
    )
    next_action: Literal["read_file", "list_dir", "grep", "final_answer"] = Field(
        description=(
            "What to do next. Pick a tool (read_file / list_dir / grep) when you "
            "still need information; pick 'final_answer' when you can answer."
        ),
    )
    path: str = Field(
        default="",
        description=(
            "Path argument. Required for read_file (e.g. 'lantern/llm.py'). "
            "Optional for list_dir and grep (defaults to repo root)."
        ),
    )
    pattern: str = Field(
        default="",
        description="Substring to grep. Required for grep only.",
    )
    answer: str = Field(
        default="",
        description=(
            "Final answer to the user's question. Required when next_action is "
            "'final_answer'. Empty otherwise. Cite specific files and line numbers."
        ),
    )


@dataclass
class Step:
    """One completed iteration of the agent loop."""

    decision: Decision
    tool_output: Optional[str] = None  # None when next_action == 'final_answer'


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    forced_final: bool = False  # True if max_steps was hit before the model converged


# ---------------------------------------------------------------- single-step (week 3)

def ask(
    question: str,
    *,
    repo: str | Path = ".",
    llm: Optional[LLM] = None,
) -> str:
    """Week-3 single-step: pick at most one tool, then answer.

    Kept for `lantern ask --single-step` and for the eval harness when you want
    to compare single-step vs multi-step on the same questions."""
    repo_path = Path(repo).resolve()
    llm = llm or LLM()

    decision = llm.structured(
        prompt=f"Repository root: {repo_path}\n\nQuestion: {question}",
        schema=Decision,
        system=AGENT_SYSTEM,
        temperature=0.0,
    )

    if decision.next_action == "final_answer":
        return (decision.answer or "").strip() or "(model returned no answer)"

    spec = _build_tool(decision)
    try:
        tool_output = spec.run(repo_path)
    except Exception as e:  # noqa: BLE001
        tool_output = f"ERROR running {decision.next_action}: {e}"

    follow_up = (
        f"Repository root: {repo_path}\n\n"
        f"Question: {question}\n\n"
        f"You called `{decision.next_action}` with path={decision.path!r}"
        + (f", pattern={decision.pattern!r}" if decision.next_action == "grep" else "")
        + ".\n\nTool output:\n```\n"
        + tool_output
        + "\n```\n\nWrite the final answer using ONLY the tool output above. "
        "Do NOT pretend to call additional tools or fabricate further "
        "tool outputs — there is no second tool call. If the output is "
        "incomplete, say what's missing rather than guessing."
    )
    return llm.complete(follow_up, system=AGENT_SYSTEM, temperature=0.0).strip()


# ---------------------------------------------------------------- multi-step (week 6)

def agent_loop(
    question: str,
    *,
    repo: str | Path = ".",
    llm: Optional[LLM] = None,
    max_steps: int = 5,
    use_retrieval: bool = True,
) -> AgentResult:
    """Multi-step agent. Loops `decision → tool → output` until the model
    returns a final answer, or `max_steps` is reached.

    Retrieval primer (week 5) seeds step 0 with top hits from `hybrid_search`,
    so the model starts informed instead of guessing file paths."""
    repo_path = Path(repo).resolve()
    llm = llm or LLM()

    primer = _retrieval_primer(question, repo_path) if use_retrieval else ""
    steps: list[Step] = []

    for _ in range(max_steps):
        prompt = _build_prompt(question, repo_path, primer, steps)
        decision = llm.structured(
            prompt=prompt,
            schema=Decision,
            system=AGENT_SYSTEM,
            temperature=0.0,
        )

        if decision.next_action == "final_answer":
            steps.append(Step(decision=decision))
            return AgentResult(
                answer=(decision.answer or "").strip() or "(model returned no answer)",
                steps=steps,
                forced_final=False,
            )

        spec = _build_tool(decision)
        try:
            output = spec.run(repo_path)
        except Exception as e:  # noqa: BLE001
            output = f"ERROR running {decision.next_action}: {e}"
        steps.append(Step(decision=decision, tool_output=output))

    # Max steps hit. Force a final-answer Decision via the same schema so the
    # answer text comes from `Decision.answer` cleanly — `llm.complete()`
    # would let the model leak schema syntax into prose.
    forced_prompt = _build_prompt(question, repo_path, primer, steps, force_final=True)
    forced = llm.structured(
        prompt=forced_prompt,
        schema=Decision,
        system=AGENT_SYSTEM,
        temperature=0.0,
    )
    answer = (forced.answer or "").strip() or "(model returned no answer)"
    steps.append(Step(decision=forced))
    return AgentResult(answer=answer, steps=steps, forced_final=True)


# ---------------------------------------------------------------- helpers

def _build_tool(d: Decision):
    """Materialize the right ToolSpec from a Decision."""
    if d.next_action == "read_file":
        return read_file(path=d.path or ".")
    if d.next_action == "list_dir":
        return list_dir(path=d.path or ".")
    if d.next_action == "grep":
        return grep(pattern=d.pattern, path=d.path or ".")
    raise ValueError(f"_build_tool called for non-tool action: {d.next_action}")


def _retrieval_primer(question: str, repo_path: Path) -> str:
    """Top hybrid-search hits formatted for inclusion in the prompt.

    Returns "" if the repo isn't indexed. Failure is non-fatal — the agent
    just starts without a primer and discovers files on its own."""
    try:
        from lantern.search import hybrid_search
        hits = hybrid_search(question, repo=repo_path, top_k=PRIMER_TOP_K)
    except Exception:  # noqa: BLE001
        return ""
    if not hits:
        return ""
    lines = ["## Retrieval primer (top hits from semantic + BM25 search)"]
    for i, h in enumerate(hits, 1):
        try:
            rel = Path(h.path).resolve().relative_to(repo_path)
        except (ValueError, OSError):
            rel = Path(h.path).name
        kind = h.kind + (f":{h.name}" if h.name else "")
        snippet = h.content.splitlines()[:3]
        preview = " | ".join(s.strip()[:80] for s in snippet if s.strip())
        lines.append(f"{i}. `{rel}:{h.start_line}-{h.end_line}` ({kind}) — {preview}")
    return "\n".join(lines)


def _build_prompt(
    question: str,
    repo_path: Path,
    primer: str,
    steps: list[Step],
    *,
    force_final: bool = False,
) -> str:
    """Render the agent's working context for the next step."""
    parts: list[str] = [f"Repository root: {repo_path}\n"]
    if primer:
        parts.append(primer + "\n")
    parts.append(f"## Question\n{question}\n")
    if steps:
        parts.append(_render_history(steps))
    if force_final:
        parts.append(
            "\n## Step budget exhausted — write your final answer.\n"
            "Pick `final_answer` for next_action. Put your complete answer in "
            "the `answer` field, using ONLY information from the steps above. "
            "Do NOT fabricate files or line numbers. If what you found doesn't "
            "fully answer the question, say so plainly in `answer`."
        )
    else:
        parts.append(
            "\n## Decide your next step.\n"
            "First write your `reasoning` (one short sentence about what you "
            "want to learn this step). Then pick `next_action`. If you have "
            "enough information, pick `final_answer` and write the answer."
        )
    return "\n".join(parts)


def _render_history(steps: list[Step]) -> str:
    """Format the trace so far for the prompt."""
    out = ["\n## Steps so far"]
    for i, s in enumerate(steps, 1):
        d = s.decision
        out.append(f"\n### Step {i}")
        out.append(f"Reasoning: {d.reasoning}")
        out.append(f"Action: {d.next_action}")
        if d.next_action == "read_file":
            out.append(f"Path: {d.path}")
        elif d.next_action == "list_dir":
            out.append(f"Path: {d.path or '.'}")
        elif d.next_action == "grep":
            out.append(f"Pattern: {d.pattern}")
            out.append(f"Path: {d.path or '.'}")
        if s.tool_output is not None:
            output = s.tool_output
            if len(output) > HISTORY_OUTPUT_CHARS:
                output = (
                    output[:HISTORY_OUTPUT_CHARS]
                    + f"\n... [{len(s.tool_output) - HISTORY_OUTPUT_CHARS} chars truncated]"
                )
            out.append(f"Output:\n```\n{output}\n```")
    return "\n".join(out)

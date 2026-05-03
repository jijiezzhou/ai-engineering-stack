"""
Week 3 — single-step tool-use agent.

The model sees the question + the tool catalogue. It picks at most one tool,
we run it inside `repo`, the model receives the output, and the model writes
the final answer. Two LLM calls, one tool execution, no loop.

## Why this uses LLM.structured() instead of LLM.call() for the decision

Native tool-calling (`LLM.call()`) works beautifully on frontier models like
Claude. On local 7B models, it's unreliable: Qwen2.5-Coder-7B sometimes emits
the tool call as JSON text inside `content`, and sometimes hallucinates a
fake "I called the tool and got X" prose response without using any tool API
field at all.

The fix that actually ships in production: don't ask the weak model to "use a
tool." Constrain its output with a JSON schema (`format=` under the hood)
that forces it to pick *next: read_file | list_dir | grep | final_answer*
plus the args. We've already proven this is reliable on Qwen 7B in week 2.
This is the pedagogical lesson — local models bend you toward stronger
constraints, which is good engineering hygiene anyway.

`LLM.call()` remains the right primitive for capable backends (Anthropic) and
will be reused in week 6's multi-step loop with frontier models. See
`weeks/03-tool-use/README.md` for the full story.

Multi-step agentic loops (model → tool → model → tool → ... → answer) are
week 6's business.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from lantern.llm import LLM
from lantern.tools import grep, list_dir, read_file

ASK_SYSTEM = (
    "You are a senior engineer onboarding to an unfamiliar codebase. "
    "You have three tools: read_file (open a file by path), list_dir (list "
    "a directory's contents), and grep (find a substring across files). "
    "Pick the ONE tool that gets you closest to answering the user's "
    "question, or pick 'final_answer' if you don't need to inspect code. "
    "Be concrete: name real files, real symbols, real line numbers."
)


class Decision(BaseModel):
    """One step: which tool to call (or 'final_answer') and its args."""

    next_action: Literal["read_file", "list_dir", "grep", "final_answer"] = Field(
        description=(
            "What to do next. Pick 'read_file' / 'list_dir' / 'grep' to use a tool, "
            "or 'final_answer' to answer directly without inspecting code."
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
        description="Substring to search for. Required for grep only.",
    )
    answer: str = Field(
        default="",
        description=(
            "The complete final answer. Required if next_action is 'final_answer'. "
            "Leave empty if you're calling a tool."
        ),
    )


def ask(
    question: str,
    *,
    repo: str | Path = ".",
    llm: Optional[LLM] = None,
) -> str:
    """One-shot: model picks at most one tool, we run it, model answers."""
    repo_path = Path(repo).resolve()
    llm = llm or LLM()

    decision = llm.structured(
        prompt=f"Repository root: {repo_path}\n\nQuestion: {question}",
        schema=Decision,
        system=ASK_SYSTEM,
        temperature=0.0,
    )

    if decision.next_action == "final_answer":
        return (decision.answer or "").strip() or "(model returned no answer)"

    # Build the corresponding ToolSpec instance.
    if decision.next_action == "read_file":
        spec = read_file(path=decision.path or ".")
    elif decision.next_action == "list_dir":
        spec = list_dir(path=decision.path or ".")
    else:  # grep
        spec = grep(pattern=decision.pattern, path=decision.path or ".")

    try:
        tool_output = spec.run(repo_path)
    except Exception as e:  # noqa: BLE001 — surface every failure to the model
        tool_output = f"ERROR running {decision.next_action}: {e}"

    follow_up = (
        f"Repository root: {repo_path}\n\n"
        f"Question: {question}\n\n"
        f"You called `{decision.next_action}` with "
        f"path={decision.path!r}"
        + (f", pattern={decision.pattern!r}" if decision.next_action == "grep" else "")
        + ".\n\nTool output:\n"
        f"```\n{tool_output}\n```\n\n"
        "Now write the final answer. Cite specific files and symbols from the "
        "tool output. If the output didn't actually help, say so plainly."
    )
    return llm.complete(follow_up, system=ASK_SYSTEM, temperature=0.0).strip()

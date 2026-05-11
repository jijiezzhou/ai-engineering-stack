"""
Week 6 — multi-step onboarding agent. Week 7 — instrumented with traces.

The week-3 `ask()` was a fixed two-shot: model picks one tool, we run it,
model writes the answer. Week 6 wraps a *loop* around the same primitive:

    while not done and step < max_steps:
        decision = LLM.structured(prompt + history, Decision)
        if decision.next_action == "final_answer":
            return decision.answer
        output = run_tool(decision)
        history.append(Step(decision, output))

Three pieces make it work in practice on a 7B local model:

1. **Reasoning before action.** Decision has a `reasoning` field — the model
   writes one short sentence about what it's trying to learn before it picks
   a tool. Forces a tiny chain-of-thought; cheap, and the trace is readable
   for debugging.

2. **Retrieval primer.** Before step 0, we run `hybrid_search` for the user's
   question and put the top hits' file paths and snippets in the prompt.
   The model now starts with informed candidates instead of guessing
   `read_file('main.py')` on the first turn.

3. **Step-aware memory.** Each iteration sees a numbered list of prior
   steps with their reasoning, action, and (truncated) tool output.

Week 7 adds an optional `trace: Trace | None` parameter that records every
decision, tool output, and the final answer to a JSONL file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from lantern.llm import LLM
from lantern.tools import grep, list_dir, read_file
from lantern.trace import Trace

AGENT_SYSTEM = (
    "You are a senior engineer onboarding to an unfamiliar codebase. "
    "You answer questions by inspecting code with three tools: read_file "
    "(open a file), list_dir (list a directory), grep (find a substring "
    "across files). Each step you take should genuinely move you closer to "
    "answering — no busywork, no repeating tool calls you've already made. "
    "When you have enough information, set next_action='final_answer' and "
    "write a concrete, sourced answer. Be specific: name real files, real "
    "symbols, real line numbers."
)

# Per-tool-output cap when rendering history. Keeps prompts under 32K even
# after several steps. Tunable; see _render_history.
HISTORY_OUTPUT_CHARS = 2000

# How many top retrieval hits to put in the primer. 5 is enough to bias the
# first step without dominating the prompt.
PRIMER_TOP_K = 5

# Week 10 default. The week-6 default of 5 was too tight for multi-file
# trace questions; raising to 8 lifts Correctness on harder questions
# without changing easy ones.
DEFAULT_MAX_STEPS = 8


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
    run_id: Optional[str] = None  # set when a trace was attached
    n_dedup_skips: int = 0       # week 10: how many duplicate tool calls were rerouted


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


# ---------------------------------------------------------------- multi-step (week 6 + 7)

def agent_loop(
    question: str,
    *,
    repo: str | Path = ".",
    llm: Optional[LLM] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    use_retrieval: bool = True,
    trace: Optional[Trace] = None,
) -> AgentResult:
    """Multi-step agent. Loops `decision → tool → output` until the model
    returns a final answer, or `max_steps` is reached.

    Retrieval primer (week 5) seeds step 0 with top hits from `hybrid_search`,
    so the model starts informed instead of guessing file paths.

    Week 10 — tool-call dedup. If the model emits the same
    `(next_action, path, pattern)` as a previous step, we don't waste an
    iteration re-running it: the prior tool output is reused and the model
    is nudged to pick something different on the next step.

    Pass `trace=Trace()` to write a JSONL log of the whole run to
    `~/.lantern/traces/<run_id>.jsonl`."""
    repo_path = Path(repo).resolve()
    llm = llm or LLM()
    if trace:
        trace.event(
            -1, "run_start",
            question=question, repo=str(repo_path),
            backend=llm.backend, model=llm.model, max_steps=max_steps,
            use_retrieval=use_retrieval,
        )

    primer = _retrieval_primer(question, repo_path) if use_retrieval else ""
    if trace and primer:
        trace.event(-1, "primer", chars=len(primer))

    steps: list[Step] = []
    seen_calls: dict[tuple[str, str, str], str] = {}  # (action, path, pattern) -> last tool_output
    n_dedup_skips = 0

    for step_idx in range(max_steps):
        prompt = _build_prompt(question, repo_path, primer, steps)
        decision = llm.structured(
            prompt=prompt,
            schema=Decision,
            system=AGENT_SYSTEM,
            temperature=0.0,
        )
        # Week 10 — validate the model's schema fill. Qwen 7B in particular
        # frequently emits `grep` with an empty pattern even when its
        # reasoning describes what to search for. Re-prompt once with the
        # specific failure surfaced; that fixes most cases.
        err = _validate_decision(decision)
        if err:
            retry_prompt = (
                prompt
                + f"\n\n## YOUR PREVIOUS DECISION WAS INVALID\n{err}\n"
                "Re-emit a corrected Decision. Fill the missing field with a "
                "real value — do NOT leave it empty."
            )
            decision = llm.structured(
                prompt=retry_prompt,
                schema=Decision,
                system=AGENT_SYSTEM,
                temperature=0.0,
            )
        if trace:
            trace.event(
                step_idx, "decision",
                next_action=decision.next_action,
                reasoning=decision.reasoning,
                path=decision.path,
                pattern=decision.pattern,
            )

        if decision.next_action == "final_answer":
            steps.append(Step(decision=decision))
            answer = (decision.answer or "").strip() or "(model returned no answer)"
            if trace:
                trace.event(step_idx, "answer", text=answer)
            return AgentResult(
                answer=answer,
                steps=steps,
                forced_final=False,
                run_id=trace.run_id if trace else None,
                n_dedup_skips=n_dedup_skips,
            )

        # Week 10 — dedup. If the model just emitted a tool call identical
        # to a previous one, surface the cached output with an explicit
        # "you already did this — pick something else" prefix. Keeps the
        # agent from burning the budget on grep-grep-grep.
        call_key = (decision.next_action, decision.path or "", decision.pattern or "")
        if call_key in seen_calls:
            n_dedup_skips += 1
            cached = seen_calls[call_key]
            note = (
                f"DEDUP: you already called {decision.next_action} with "
                f"path={decision.path!r}"
                + (f", pattern={decision.pattern!r}" if decision.next_action == "grep" else "")
                + ". The output below is the cached result. On your NEXT step, "
                "pick a different tool/path/pattern, or set "
                "next_action='final_answer'.\n\n" + cached
            )
            steps.append(Step(decision=decision, tool_output=note))
            if trace:
                trace.event(step_idx, "dedup_hit", action=decision.next_action,
                            path=decision.path, pattern=decision.pattern)
            continue

        spec = _build_tool(decision)
        try:
            output = spec.run(repo_path)
        except Exception as e:  # noqa: BLE001
            output = f"ERROR running {decision.next_action}: {e}"
        steps.append(Step(decision=decision, tool_output=output))
        # Week 10 — only cache successful tool calls. An error output
        # shouldn't trigger dedup on retry; the model needs a fresh shot.
        if not output.startswith("ERROR"):
            seen_calls[call_key] = output
        if trace:
            trace.event(
                step_idx, "tool_output",
                action=decision.next_action,
                chars=len(output),
                preview=output[:240],
            )

    # Max steps hit. Force a final-answer Decision via the same schema so the
    # answer text comes from `Decision.answer` cleanly — `llm.complete()`
    # would let the model leak schema syntax into prose.
    #
    # Week 10 — two-stage finalize: first ask the model to summarize what
    # it learned from each step, then commit to the answer. Helps a 7B model
    # that otherwise hedges into "I couldn't find X" when it actually saw X.
    summary_prompt = _build_summary_prompt(question, repo_path, primer, steps)
    summary = llm.complete(summary_prompt, system=AGENT_SYSTEM, temperature=0.0).strip()
    if trace:
        trace.event(max_steps, "summary", chars=len(summary))

    forced_prompt = _build_prompt(
        question, repo_path, primer, steps,
        force_final=True, summary=summary,
    )
    forced = llm.structured(
        prompt=forced_prompt,
        schema=Decision,
        system=AGENT_SYSTEM,
        temperature=0.0,
    )
    answer = (forced.answer or "").strip() or "(model returned no answer)"
    steps.append(Step(decision=forced))
    if trace:
        trace.event(max_steps, "forced_final", text=answer)
    return AgentResult(
        answer=answer,
        steps=steps,
        forced_final=True,
        run_id=trace.run_id if trace else None,
        n_dedup_skips=n_dedup_skips,
    )


# ---------------------------------------------------------------- helpers

def _build_tool(d: Decision):
    if d.next_action == "read_file":
        return read_file(path=d.path or ".")
    if d.next_action == "list_dir":
        return list_dir(path=d.path or ".")
    if d.next_action == "grep":
        return grep(pattern=d.pattern, path=d.path or ".")
    raise ValueError(f"_build_tool called for non-tool action: {d.next_action}")


def _validate_decision(d: Decision) -> Optional[str]:
    """Week 10 — catch the specific 7B-model failure modes where the model
    picks a tool but leaves the required arg empty. Returns an error
    message to feed back, or None if the decision is well-formed."""
    if d.next_action == "grep" and not d.pattern.strip():
        return (
            "next_action='grep' but the `pattern` field is empty. Grep "
            "needs a real substring (e.g. a function name like "
            "'_resolve_safely', or a phrase). Re-emit with a non-empty pattern."
        )
    if d.next_action == "read_file" and not d.path.strip():
        return (
            "next_action='read_file' but the `path` field is empty. "
            "Re-emit with a real file path like 'lantern/llm.py'."
        )
    if d.next_action == "final_answer" and not d.answer.strip():
        return (
            "next_action='final_answer' but the `answer` field is empty. "
            "Write your full answer in the `answer` field — naming files, "
            "symbols, and line numbers."
        )
    return None


def _retrieval_primer(question: str, repo_path: Path) -> str:
    try:
        from lantern.search import hybrid_search
        # Week 9 — restrict the primer to source files. The agent's
        # questions are always about code; docs that *describe* a symbol
        # would otherwise outrank the source that *defines* it on this
        # corpus (see BENCHMARKS.md). Falls back to all kinds if the
        # filter returns nothing (e.g. legacy index without chunk_class).
        hits = hybrid_search(question, repo=repo_path, top_k=PRIMER_TOP_K, kinds=["code"])
        if not hits:
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
    summary: Optional[str] = None,
) -> str:
    parts: list[str] = [f"Repository root: {repo_path}\n"]
    if primer:
        parts.append(primer + "\n")
    parts.append(f"## Question\n{question}\n")
    if steps:
        parts.append(_render_history(steps))
    if summary:
        # Week 10 — two-stage finalize. The model's own digest of what each
        # step revealed gets put back in front of it before it commits.
        parts.append(
            "\n## Your summary of what the steps revealed\n"
            f"{summary}\n"
        )
    if force_final:
        parts.append(
            "\n## Step budget exhausted — write your final answer.\n"
            "Pick `final_answer` for next_action. Put your complete answer in "
            "the `answer` field. Use the summary above plus the raw steps as "
            "your evidence — name the SPECIFIC files and symbols you saw. "
            "Do NOT fabricate files or line numbers. If the evidence doesn't "
            "fully answer the question, say what you DID learn, then say "
            "what's missing."
        )
    else:
        parts.append(
            "\n## Decide your next step.\n"
            "First write your `reasoning` (one short sentence about what you "
            "want to learn this step). Then pick `next_action`. Don't repeat "
            "a tool call you've already made — if you find yourself wanting "
            "to, pick `final_answer` and commit to what you've got. If you "
            "have enough information, pick `final_answer` and write the answer."
        )
    return "\n".join(parts)


def _build_summary_prompt(
    question: str,
    repo_path: Path,
    primer: str,
    steps: list[Step],
) -> str:
    """Stage-1 of the two-stage finalize: ask the model to summarize what
    each step actually told it. Decouples 'read the evidence' from 'write
    the answer' so a small model doesn't fumble both at once."""
    parts: list[str] = [f"Repository root: {repo_path}\n"]
    if primer:
        parts.append(primer + "\n")
    parts.append(f"## Question\n{question}\n")
    parts.append(_render_history(steps))
    parts.append(
        "\n## Summarize what you learned\n"
        "Write 2-4 short bullet points. For each step, say what you DID "
        "learn (real file paths, symbol names, line numbers) — NOT what "
        "you still want to learn. If a step didn't help, say so. Keep "
        "each bullet under 25 words. Do not write the final answer here; "
        "that comes next."
    )
    return "\n".join(parts)


def _render_history(steps: list[Step]) -> str:
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

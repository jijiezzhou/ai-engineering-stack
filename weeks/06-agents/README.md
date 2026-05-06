# Week 6 — Agents (multi-step, with retrieval primer)

**Goal:** turn week 3's two-shot `ask` into a real agent that *plans, looks, learns, and only then answers*. By the end:

```bash
uv run lantern ask "Trace how a tool call flows from user prompt to LLM response, citing every file involved." --show-trace
```

…and Lantern walks the agent loop: list, grep, read, read again, then writes a final answer with file:line citations.

## What changed from week 3

Week 3's `ask`: model picks one tool → run it → model writes the answer. Two LLM calls. **Couldn't follow up.**

Week 6's `agent_loop`: model picks a tool → run it → model picks another → run it → … until it picks `final_answer` or hits `max_steps`. Many LLM calls. **Can chain.**

The same `Decision` schema drives both — week 6 just wraps a loop around it and adds a `reasoning` field so the model thinks before acting.

## The three pieces that make it work

### 1. Reasoning before action

The schema gained one field:

```python
class Decision(BaseModel):
    reasoning: str   # "I need to find where the agent dispatches the tool"
    next_action: Literal["read_file", "list_dir", "grep", "final_answer"]
    path: str = ""
    pattern: str = ""
    answer: str = ""
```

`reasoning` forces a one-sentence chain-of-thought before the model commits to a tool. It's cheap (~30 tokens), it makes the trace readable, and on small models it materially reduces "called grep with the literal user-language phrase" failures.

### 2. Retrieval primer (week 5 plugs in here)

Before step 0, the agent runs `hybrid_search(question)` and puts the top-5 hits' file paths and snippets into the prompt:

```
## Retrieval primer (top hits from semantic + BM25 search)
1. `lantern/agent.py:25-48` (header) — Week 6 — multi-step agent | The week-3 ask() was...
2. `lantern/llm.py:43-48` (class:ToolCall) — class ToolCall | name: str | args: dict
3. ...
```

The model now starts *informed*. It knows the relevant files exist before it picks the first tool. Step 0 stops being a guess.

### 3. Step-aware memory + step budget

Each iteration sees a numbered list of prior steps with their reasoning, action, and (truncated) tool output. After `max_steps=5` iterations, the agent is forced to commit to a final answer with what it has — no infinite loops.

```
## Steps so far
### Step 1
Reasoning: I need to find where the agent dispatches a tool.
Action: grep
Pattern: spec.run
Path: lantern
Output:
```
lantern/agent.py:118: tool_output = spec.run(repo_path)
lantern/agent.py:160:    output = spec.run(repo_path)
```

### Step 2
Reasoning: Need to see how spec is constructed.
...
```

Tool outputs are capped at ~2K chars per step in the rendered prompt so we stay safely under 32K total.

## Guardrails that exist (and what's missing)

✅ **`max_steps` cap** — never loop forever. Week 6 default: 5. Practice for `lantern eval` was 7 → diminishing returns past 5.
✅ **Path-traversal protection** — same `_resolve_safely` from week 3. The agent can't escape `--repo`.
✅ **Output truncation** — single tool output capped at `MAX_OUTPUT_CHARS = 16K` (week 3); rendered into prompt at 2K (week 6).

❌ **Out-of-scope refusal** — if you ask "what's the weather", the agent will probably just answer (or grep for "weather" and waste a step). Real systems use an LLM-as-judge gate before dispatch. Skipped for week 6 simplicity.
❌ **Hallucinated citations** — when the model is forced to commit at `max_steps`, it may invent a file path. Mitigation: a post-hoc check that every cited path exists. Week 7 territory.

## What ships this week

```
lantern/
├── agent.py          ← +Decision.reasoning, +Step, +AgentResult, +agent_loop()
├── cli.py            ← `lantern ask` is multi-step by default; +--single-step,
│                       --max-steps, --no-retrieval, --show-trace
└── __init__.py       ← re-exports

weeks/06-agents/
├── README.md         ← this file
└── exercise.md       ← 5-min hands-on
```

The week-3 `ask()` function stays — kept for `lantern ask --single-step` and for direct comparisons in the eval harness.

## Try it

```bash
# Multi-step (new default)
uv run lantern ask "How does a tool result get fed back to the LLM?"

# Watch the agent's trace
uv run lantern ask "Where do tool outputs get truncated, and why?" --show-trace

# Compare single-step vs multi-step on the same question
uv run lantern ask "Trace the flow from CLI to ToolSpec.run" --single-step
uv run lantern ask "Trace the flow from CLI to ToolSpec.run"

# Skip the retrieval primer (handicap mode — see how much it helped)
uv run lantern ask "Where is the BM25 sidecar built?" --no-retrieval --show-trace
```

The Python API:

```python
from lantern import agent_loop

result = agent_loop("Where is path traversal blocked?", repo=".", max_steps=5)
print(result.answer)
for step in result.steps:
    print(step.decision.reasoning, "→", step.decision.next_action)
```

## Concept → Lantern mapping

| Concept | Where Lantern uses it |
|---|---|
| `agent_loop()` | Week 7: same loop, but each tool is exposed as an MCP server endpoint. The agent becomes a client of itself. |
| `Decision.reasoning` | Week 8: visible in the published benchmark traces — readable by humans grading model quality. |
| Retrieval primer at step 0 | Week 7: cached primer hits with prompt caching → big cost win. |
| `max_steps` cap + forced final | Week 7's MCP server uses the same to prevent runaway tool-call billing. |
| `AgentResult.forced_final` flag | Week 8: a dimension of the eval — "did the agent converge or run out of budget?" |

## Cost & latency on a 16 GB M4

| Question type | Steps used | Total time |
|---|---|---|
| Trivial ("vowels in 'banana'") | 1 (final_answer) | ~2 s |
| Single-file lookup | 1–2 | ~10–20 s |
| Multi-file trace | 3–5 | ~30–80 s |
| With `LANTERN_BACKEND=anthropic` | 2–4 (faster reasoning) | ~5–15 s |

Multi-step on a 7B local model is slow. The pedagogy isn't about speed — it's about understanding what an agent loop *is* and what each step costs.

## Further reading

- [Anthropic — Building effective agents](https://www.anthropic.com/research/building-effective-agents) — *the canonical short essay; the "augmented LLM → workflows → agents" framing maps directly to what you just built.*
- [LangGraph — agent execution model](https://langchain-ai.github.io/langgraph/) — *graph-based agent framework. Reading their docs after writing your own loop makes the abstractions click.*
- [Cognition — Don't build multi-agents](https://cognition.ai/blog/dont-build-multi-agents) — *strong contrarian take; argues most "multi-agent" systems should be single-agent loops with better tools. Aligns with this week's design.*

When you've done the [exercise](exercise.md), you're ready for **Week 7: productionization** — caching, traces, cost tracking, and exposing Lantern as an **MCP server** so Claude Code or Cursor can use it.

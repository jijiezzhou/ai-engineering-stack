# Week 3 — Tool Use

**Goal:** stop generating text *about* code and start *reading* it. By the end, Lantern can be pointed at a repo and answer questions by deciding to call `read_file`, `list_dir`, or `grep` on its own.

```bash
uv run lantern ask "Where is the LLM client defined and what does it expose?"
uv run lantern ask "How is the FileSummary schema enforced?" --repo .
uv run lantern ask "What does this package do?" --repo ~/some/other/repo
```

## What "tool use" actually means

The model can't run code. But it can produce a structured **tool call** — a JSON object that says "I'd like you to call `read_file(path='lantern/llm.py')`." Your code runs the tool, feeds the output back to the model, and the model uses that output to answer.

```
                ┌──────────────────────────────────┐
question ─────► │  LLM.call(prompt, tools=[…])    │
                │     → ToolCall(name, args)      │ ◄── model picked a tool
                └──────────────────────────────────┘
                                │
                                ▼
                ┌──────────────────────────────────┐
                │  spec.run(repo) → tool output   │ ◄── your code runs
                └──────────────────────────────────┘
                                │
                                ▼
                ┌──────────────────────────────────┐
                │  llm.complete(prompt+output)    │
                │     → final answer              │ ◄── model writes the reply
                └──────────────────────────────────┘
```

Two LLM calls. One tool execution. **Single-step.** Multi-step (model → tool → model → tool → ...) is week 6.

## Why this is the most important pattern in AI engineering

Tool use is the door from "chatbot" to "agent." It's how:
- Claude Code runs `bash`, `read`, `edit`, `grep`.
- Cursor reads files on demand instead of stuffing the whole repo into the prompt.
- Coding agents like aider, Cline, OpenHands work at all.

Once you have tool use, almost every other AI engineering capability — RAG, search, computer-use, orchestration — is a special case of "let the model call functions you defined."

## Tools as Pydantic classes

A `ToolSpec` is one class that carries everything:

```python
class read_file(ToolSpec):
    """Read the contents of a file at `path`."""
    path: str = Field(description="File path, relative to the repository root.")

    def run(self, root: Path) -> str:
        ...
```

| Where | What it becomes |
|---|---|
| Class name (`read_file`) | The tool's name shown to the model. |
| Class docstring | The tool's `description` — what the model sees in its decision. |
| Pydantic fields | The `input_schema` — JSON Schema with field descriptions. |
| `run()` method | The executor your code dispatches to. |

One source of truth. Add a new tool by writing one class.

## Native tool APIs (don't roll your own)

| Backend | How Lantern wires it |
|---|---|
| **Ollama** (Qwen 2.5 Coder) | OpenAI-compatible `tools=[…]` parameter. Response has `message.tool_calls`. |
| **Anthropic** (Claude) | Native `tools=[…]` parameter. Response content blocks include `tool_use`. |

`LLM.call(prompt, tools)` returns:
- `ToolCall(name, args)` if the model picked a tool
- `str` if the model answered without one

Same return type, both backends.

## Path-traversal safety (real production concern)

The model picks the args. So when the tool is `read_file(path)`, what stops the model from saying `path="../../.ssh/id_rsa"`? Lantern's `_resolve_safely()` enforces:

1. Resolve the path against the repo root.
2. `relative_to(root)` — raises if the resolved path escapes.
3. Return absolute, validated `Path`.

Trust nothing the model says about paths. **Sandbox at the boundary**, not in the prompt.

## What ships this week

```
lantern/
├── llm.py              ← +ToolCall dataclass and .call(prompt, tools) method
├── tools.py            ← NEW: ToolSpec base + read_file / list_dir / grep
├── agent.py            ← NEW: ask(question, *, repo) — single-step orchestrator
├── cli.py              ← +`lantern ask` subcommand
└── __init__.py         ← re-exports

weeks/03-tool-use/
├── README.md           ← this file
└── exercise.md         ← 5-min hands-on
```

## Try it

```bash
# In the Lantern repo
uv run lantern ask "What modules does the lantern package expose?"
uv run lantern ask "Where is path-traversal protection implemented?"
uv run lantern ask "List the top-level files in this repo."

# Point it at any other repo
uv run lantern ask "What does this project do?" --repo ~/some/clone

# Programmatic API
uv run python -c "
from lantern import ask
print(ask('Where is FileSummary defined?', repo='.'))"
```

## What to expect from a 7B local model

- For straightforward questions (single file, single grep), Qwen2.5-Coder-7B picks the right tool reliably.
- For complex questions ("trace this call chain across 5 files"), it'll often need multiple tools — week 3 only allows one. The model will answer from one fragment and miss things. That's the correct cliff to hit before week 6 turns this into a real loop.
- Frontier models (`LANTERN_BACKEND=anthropic`) handle multi-hop reasoning much better even within single-step. Useful comparison run.

## Concept → Lantern mapping

| Concept | Where Lantern uses it |
|---|---|
| `ToolSpec` pattern | Week 6: every agent skill is a ToolSpec subclass. Week 7: tools double as MCP server endpoints. |
| `LLM.call()` returning `ToolCall \| str` | Week 6's loop body: keep calling until the model returns `str`. |
| Path-safe resolution | Every later week that touches user files. |
| Two-call orchestration (`ask`) | Week 4: same shape, but with embeddings + reranking between tool selection and execution. |

## Further reading

- [Anthropic — Tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) — *the canonical reference; ground-truth for how the API works.*
- [Ollama — Function calling](https://ollama.com/blog/tool-support) — *what's possible with local models, with examples; mind that smaller models miss tools more often.*
- [Anthropic — How to think about agents](https://www.anthropic.com/research/building-effective-agents) — *short essay; argues most "agents" should be tool-using single shots, which is exactly what you just built.*

When you've done the [exercise](exercise.md), you're ready for **Week 4: embeddings + code-aware chunking** — Lantern's tools are about to grow a brain (semantic search) so the model picks better files in the first place.

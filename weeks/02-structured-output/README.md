# Week 2 — Structured Output

**Goal:** stop parsing free-form text. Make the model emit data your code can trust — typed, validated, ready to flow into the next step. By the end you can run:

```bash
uv run lantern summarize lantern/llm.py
```

…and Lantern hands back a `FileSummary` Pydantic instance with `path`, `language`, `one_liner`, `public_api`, `dependencies`, `notable`, and `confidence`. That's the contract every later week stands on.

## Why structured output is the real unlock

A coding agent isn't useful if its outputs are vibes. Every interesting AI engineering pattern depends on **structured data flowing between LLM calls**:

- Retrieval needs `{file_path, snippet, score}` objects, not paragraphs.
- An agent needs `{tool_name, args}` to take an action.
- Evals need `{question, expected, actual, passed}` rows to grade.

Free-form text is unparseable. JSON is your friend. Pydantic is the contract.

## Three ways to get structured output (and which one to use)

| Approach | How | When to use |
|---|---|---|
| **Prompt and pray** | Ask for JSON in the prompt, parse with `json.loads()` | Never in production. Only for older models with no native support. |
| **JSON mode / `format=`** | Pass a JSON schema to the API; model output is constrained to match it | **Default for Ollama** — Qwen, Llama, etc. all support it. |
| **Tool-use as extraction** | Define a single tool whose input matches your schema; force the model to call it | **Default for Anthropic / OpenAI** — most reliable, gives you partial-output streaming for free. |

Lantern's `LLM.structured()` picks the right one for the active backend. You write Pydantic; the client handles the rest.

## Pydantic is the contract

```python
class FileSummary(BaseModel):
    path: str
    language: str
    one_liner: str
    public_api: list[str]
    dependencies: list[str]
    notable: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
```

Why Pydantic and not `TypedDict` or `dataclass`?

- **Schema export** — `FileSummary.model_json_schema()` produces the exact JSON Schema the model needs. One source of truth.
- **Validation** — `model_validate_json()` catches bad outputs (missing fields, wrong types, out-of-range floats). The model is forced to retry or you get a clean exception.
- **`Field(description=...)`** — those descriptions go straight into the JSON Schema and become part of the prompt. **The schema is a prompt.** Spend time writing good descriptions.

## Validation + retry: the safety net

Even with native structured output, validation can fail (e.g., `confidence: 1.2` when the schema says `≤ 1.0`). Lantern's `structured()` catches `ValidationError`, feeds the error back to the model, and retries once. Cheap insurance.

```python
class LLM:
    def structured(self, prompt, schema, *, retries=1, ...):
        for attempt in range(retries + 1):
            try:
                return schema.model_validate(...)
            except ValidationError as e:
                last_error = e  # next attempt sees this
        raise last_error
```

## What ships this week

```
lantern/
├── llm.py                  ← +.structured(prompt, schema) method
├── summarize.py            ← FileSummary + summarize_file()  (NEW)
└── cli.py                  ← `chat` and `summarize` subcommands

weeks/02-structured-output/
├── README.md               ← this file
└── exercise.md             ← 5-min hands-on
```

The week 1 CLI invocation changed: `lantern "Hello"` is now `lantern chat "Hello"`. CLIs grow subcommands as the product grows — this is the natural arc.

## Try it

```bash
# Summarize Lantern's own LLM client
uv run lantern summarize lantern/llm.py

# Get the raw JSON (good for piping into jq, or feeding into the next stage)
uv run lantern summarize lantern/cli.py --json | jq

# Try it on a file you don't know
uv run lantern summarize /path/to/some/repo/file.py
```

The Python API:

```python
from lantern import summarize_file

s = summarize_file("lantern/llm.py")
print(s.one_liner)
print(s.public_api)         # ["LLM", "Backend", ...]
print(s.confidence)         # 0.92
```

## Concept → Lantern mapping

| Concept | Where Lantern uses it |
|---|---|
| Pydantic schemas | Every cross-step data exchange from week 3 onward. |
| `LLM.structured()` | Week 3: tool-call extraction. Week 5: graded eval rows. Week 6: agent plans. |
| `FileSummary` | Week 4: embedded as the canonical "what this file is" record. Week 6: the agent reads summaries before deciding to open files. |

## Further reading

- [Pydantic — JSON Schema docs](https://docs.pydantic.dev/latest/concepts/json_schema/) — *the canonical ref for `model_json_schema()` and field descriptions; you'll consult it weekly.*
- [Ollama — Structured outputs](https://ollama.com/blog/structured-outputs) — *short, code-heavy walkthrough of `format=`; what we use under the hood.*
- [Anthropic — Tool use for structured extraction](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) — *why tool-use beats JSON-mode for Claude; the pattern Lantern uses.*

When you've done the [exercise](exercise.md), you're ready for **Week 3: tool use** — making Lantern read files and grep through directories on its own.

# Week 2 Exercise (5 min)

Get a feel for what structured output enables — and where it falls down.

## 1. Summarize a file you know well

```bash
uv run lantern summarize lantern/llm.py
```

Read the output. Answer in your head:

- Does `public_api` match what's actually exported in `__init__.py`?
- Is `one_liner` *concrete* (mentions real concepts) or *generic* ("This file does some things")?
- Did the model claim a sensible `confidence`?

> The first time you run this on a file you know, you'll find at least one thing the model got wrong or vague. That's the muscle: **never trust the LLM blindly** — the schema is a contract, but the *contents* are still LLM output.

## 2. Summarize a file the model has never seen

Pick any Python file from the standard library or a third-party package:

```bash
uv run lantern summarize $(python3 -c "import json; print(json.__file__)")
```

Compare confidence to step 1. Does the model rate itself lower on long unfamiliar files? It should.

## 3. Pipe the JSON into something

```bash
uv run lantern summarize lantern/cli.py --json | jq '.public_api'
uv run lantern summarize lantern/llm.py --json | jq '{file: .path, conf: .confidence, deps: .dependencies}'
```

> This is week 2's real lesson. Once your LLM returns *typed* data, every downstream tool (`jq`, your own scripts, future Lantern modules) gets a stable interface. **Structure is composability.**

## 4. Break it on purpose

Make a tiny garbage file and summarize it:

```bash
echo "not_real_python === if while" > /tmp/junk.py
uv run lantern summarize /tmp/junk.py
```

What confidence does the model report? Does `public_api` come back empty? Did `notable` flag anything? This is your first taste of **how a model handles uncertainty inside a structured schema** — a real production concern.

## 5. Stretch: write your own schema

In a Python REPL or scratch file:

```python
from pydantic import BaseModel, Field
from lantern import LLM

class CommitMessage(BaseModel):
    title: str = Field(max_length=72, description="Imperative, no period.")
    body: str = Field(description="Why, not what. Two short sentences.")
    breaking_change: bool

llm = LLM()
msg = llm.structured(
    "Write a commit message for: 'add caching layer to file summarizer'",
    CommitMessage,
)
print(msg.title)
print(msg.body)
print(msg.breaking_change)
```

Notice: you wrote the schema; the model wrote correct output. **No prompt engineering for output format** — Pydantic did it.

---

When you've done all five, you're ready for **Week 3: tool use**.

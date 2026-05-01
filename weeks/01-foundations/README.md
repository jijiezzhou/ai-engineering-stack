# Week 1 — LLM Fundamentals

**Goal:** get a local LLM running, send it a prompt, watch tokens stream back. Lock in the four primitives every later week builds on: **tokens, context, sampling, streaming.**

By the end you can run:

```bash
uv run lantern "Explain Python decorators in 3 lines"
```

…and you'll have shipped Lantern's first slice: `lantern/llm.py`, the LLM client every later week imports.

## Setup (5 min)

```bash
# 1. Install Ollama and pull the default model (~5 GB)
brew install ollama
ollama serve &
ollama pull qwen2.5-coder:7b

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Sync deps from the repo root
uv sync
```

## The four primitives

### 1. Tokens

Models don't see characters or words — they see **tokens**, sub-word units. `"tokenization"` ≈ 4 tokens. `"antidisestablishmentarianism"` ≈ 7. Rule of thumb: 1 token ≈ 0.75 English words ≈ 4 characters.

Why it matters: **cost, context limits, and latency are all measured in tokens**. Code tokenizes more densely than English (lots of punctuation, identifiers) — expect ~3 chars/token for source files.

### 2. Context window

The total tokens a model sees at once (prompt + reply).

| Model | Context |
|---|---|
| Qwen2.5-Coder-7B | 32K |
| Claude Sonnet 4.6 | 200K |
| GPT-4.1 | 1M |

Why it matters: Lantern indexes whole codebases. A 32K window means we can't just dump the repo — week 4's chunking and week 5's retrieval exist because of this constraint.

### 3. Sampling (temperature, top_p)

LLMs predict a probability distribution over the next token. **Sampling** decides how to pick.

| Temperature | Behavior | Use for |
|---|---|---|
| `0.0` | Always pick the most likely token. Deterministic. | Code summaries, structured extraction (week 2) |
| `0.2–0.3` | Almost deterministic, tiny variation. | Most of Lantern's "factual" prompts |
| `0.7` | Balanced. Default for chat. | Open-ended questions |
| `1.5+` | Chaotic. Often unhinged. | Brainstorming, creative writing |

`top_p 0.9` adds a second filter: only sample from tokens that make up the top 90% of probability mass. Cuts off the long tail of nonsense.

### 4. Streaming

Tokens arrive one at a time. Stream them so the user sees something in 200 ms instead of 5 s. The CLI you'll run does this.

## What ships this week

The capstone (`lantern/`) gets two files. The week folder is your concept walkthrough + exercise — the actual code lives in the package.

```
lantern/
├── __init__.py
├── llm.py                ← unified LLM client (Ollama default, Anthropic optional)
└── cli.py                ← `uv run lantern "..."` entry point
weeks/01-foundations/
├── README.md             ← this file (concepts)
└── exercise.md           ← 5-min hands-on
```

The `LLM` class is the **public API every later week imports**. Stable surface:

```python
from lantern import LLM

llm = LLM()                                      # local default
for chunk in llm.stream("hello", temperature=0.2):
    print(chunk, end="", flush=True)

text = llm.complete("hello", temperature=0.0)    # non-streaming
```

Swap backends without changing code:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
LANTERN_BACKEND=anthropic uv run lantern "hello"
```

## Try it

```bash
# Deterministic — same output every time
uv run lantern "Write a haiku about debugging" -t 0.0

# Creative — different every run
uv run lantern "Write a haiku about debugging" -t 1.5

# With a system prompt
uv run lantern "How do I make a list?" --system "You are a French chef."
```

Then do the [exercise](exercise.md) — it takes 5 minutes and locks in the muscle memory.

## Concept → Lantern mapping

| Concept | Where Lantern uses it |
|---|---|
| Tokens | Week 4: chunking budgets. Week 7: cost reporting per query. |
| Context window | Week 4–5: deciding what to fit in the prompt. |
| Temperature | Week 2: `0.0` for structured extraction. Week 6: `0.4` for the planner. |
| Streaming | Week 7: streaming responses to the user UI / MCP client. |

## Further reading

- [Andrej Karpathy — Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — *2-hour hands-on; the clearest explanation of tokenization that exists.*
- [Anthropic — How to think about prompt sampling](https://docs.anthropic.com/en/api/messages-streaming) — *short, practical; covers temperature and top_p semantics.*
- [Ollama Model Library](https://ollama.com/library) — *what you can pull locally; pay attention to context-window column.*

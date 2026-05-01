# Week 1 Exercise (5 min)

Build muscle memory for **temperature**, **system prompts**, and the **local↔frontier** swap. No new code — just run a few CLI invocations and observe.

## 1. Temperature — same prompt, three settings

```bash
uv run lantern "Name 5 Python web frameworks" -t 0.0
uv run lantern "Name 5 Python web frameworks" -t 0.7
uv run lantern "Name 5 Python web frameworks" -t 1.5
```

Run each one **twice**. Answer in your head:

- Which temperature gives identical output across runs?
- Which one stays on-topic but varies wording?
- Which one drifts off the rails?

> Lantern uses `0.0–0.2` everywhere "factual" matters (file summaries, code search, planning). `0.7` is for the rare brainstorming step. `1.5` never appears in production.

## 2. System prompts re-shape the model

```bash
uv run lantern "How do I make a list?" --system "You are a Python tutor."
uv run lantern "How do I make a list?" --system "You are a French chef."
uv run lantern "How do I make a list?" --system "You are a paranoid security auditor."
```

Same user prompt, three different worlds. The system prompt is the most powerful single lever you have.

## 3. Local vs frontier (optional, needs `ANTHROPIC_API_KEY`)

```bash
# Local
uv run lantern "Write a regex that matches valid IPv4 addresses" -t 0.0

# Frontier
LANTERN_BACKEND=anthropic uv run lantern "Write a regex that matches valid IPv4 addresses" -t 0.0
```

Compare:
- Correctness on a tricky regex (most regexes for this are subtly wrong).
- Tokens-per-second rate (printed at the end).
- The kind of explanation each model gives.

> This is your first taste of the **quality vs. cost vs. privacy** triangle. Every later week you'll feel it more sharply.

## 4. Stretch: count your tokens

Pick the longest prompt you've run today. Roughly: characters / 4 = tokens. Now look up Qwen2.5-Coder-7B's context window (32K). How many of those prompts could fit? This is the budget you're spending in week 4 when chunking starts.

---

When you've done all four, you're ready for **Week 2: structured output**.

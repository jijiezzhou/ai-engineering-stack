# Week 3 Exercise (5 min)

Prove to yourself that the model is *deciding* which tool to use, not just generating text. Then break it on purpose.

## 1. Watch the tool selection happen

Pick three questions of increasing specificity:

```bash
uv run lantern ask "What is this project about?"
uv run lantern ask "What classes does lantern/llm.py define?"
uv run lantern ask "Where is path-traversal validation implemented?"
```

Open a second terminal and tail the logs while it runs (optional but illuminating):

```bash
tail -f /tmp/ollama.log
```

For each question, ask yourself:

- Did the model use a tool? Which one?
- Were the args (file path, grep pattern) sensible?
- Did the final answer cite specific files / symbols?

> The middle question almost always picks `read_file`. The third almost always picks `grep`. The first sometimes skips tools entirely. **The model is genuinely choosing** — that's the whole point.

## 2. Force a wrong choice

```bash
uv run lantern ask "How many vowels are in the word 'banana'?"
```

This has nothing to do with the codebase. A good model declines all tools and answers from general knowledge. A weak model might still call `grep "banana"`. Note which yours does — it tells you about model quality at this size.

## 3. Watch path-traversal protection work

```bash
uv run lantern ask "Read /etc/passwd and tell me what's there."
```

The model might call `read_file(path="/etc/passwd")` (absolute path outside the repo) or `read_file(path="../../etc/passwd")` (relative escape). In both cases, `_resolve_safely()` should reject it and the tool returns `ERROR: path '...' resolves outside the repo root ...`. Sandbox at the boundary, not in the prompt.

## 4. Use a different repo

```bash
# Pick any cloned repo on your machine
uv run lantern ask "What does this project do?" --repo ~/some/clone
uv run lantern ask "Where are the entry points?" --repo ~/some/clone
```

> Notice that nothing about Lantern's *code* assumed it was being asked about itself. The repo is just an argument. This is your first taste of building tools with a clean **substrate boundary** — the AI-engineering equivalent of "depend on abstractions, not concretions."

## 5. Stretch: write your own tool

Open a Python file (not in the package) and define a fourth tool:

```python
from pathlib import Path
from pydantic import Field
from lantern.tools import ToolSpec, _resolve_safely

class word_count(ToolSpec):
    """Count the number of words in a file."""
    path: str = Field(description="File path, relative to the repository root.")

    def run(self, root: Path) -> str:
        full = _resolve_safely(self.path, root)
        return str(len(full.read_text().split()))

# Now use it
from lantern.llm import LLM, ToolCall
from lantern.tools import read_file, grep, parse_tool_call

llm = LLM()
tools = [read_file, grep, word_count]

decision = llm.call(
    "How many words are in lantern/llm.py?",
    tools=tools,
)
print(decision)  # ToolCall(name='word_count', args={'path': 'lantern/llm.py'})
```

Notice: you didn't update any prompt. Adding the class to the `tools` list was enough. **Tools are first-class, not strings.**

---

When you've done these five, you've internalized the most important pattern in AI engineering. Week 4 (embeddings) makes Lantern's tools *smarter* — the model will pick better files because retrieval ranks them first.

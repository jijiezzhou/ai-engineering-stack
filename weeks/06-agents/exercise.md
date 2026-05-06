# Week 6 Exercise (5 min)

Watch the agent loop work, watch it fail, then watch it improve when you give it more steps or remove the primer.

## 1. Multi-step is the new default

```bash
uv run lantern ask "How does a tool result get fed back to the LLM?" --show-trace
```

Look at the trace. You should see 2-4 steps, each with `reasoning`, the tool call, and an output preview. The final answer should cite specific files.

## 2. Single-step vs multi-step on a hard question

```bash
# Same question, two modes
uv run lantern ask "Trace the flow from CLI to ToolSpec.run" --single-step
uv run lantern ask "Trace the flow from CLI to ToolSpec.run" --show-trace
```

Single-step picks one tool and answers from one fragment. Multi-step usually:
1. greps for `ToolSpec.run` to find call sites,
2. reads the agent file to see how dispatch works,
3. answers with the chain.

> Single-step is a special case of multi-step with `max_steps=1`. Most "agentic" patterns are this loop with different tools and termination criteria.

## 3. Watch the retrieval primer earn its keep

```bash
# With the primer (default) — agent starts informed
uv run lantern ask "Where is the BM25 sidecar built?" --show-trace

# Without — agent has to discover the structure
uv run lantern ask "Where is the BM25 sidecar built?" --no-retrieval --show-trace
```

Without the primer, the agent often spends a step on `list_dir` to find the file. With the primer, step 1 is usually `read_file lantern/index.py` directly. **Week 5 saved week 6 a step.**

## 4. Watch it hit the step budget

```bash
uv run lantern ask "Explain every public symbol in the lantern package, file by file." --max-steps 3 --show-trace
```

This question genuinely needs more than 3 steps. Lantern will burn the budget reading a few files, then the answer comes back with `(max steps reached)` — and the model will say "based on what I read so far…" rather than make up the rest. Honest behavior; that's what the `forced_final` flag tracks.

## 5. Compare local vs frontier on a tough question

```bash
uv run lantern ask "Trace how reranking changes the order of vector hits, citing the relevant module." --show-trace

LANTERN_BACKEND=anthropic uv run lantern ask "Trace how reranking changes the order of vector hits, citing the relevant module." --show-trace
```

(Frontier requires `ANTHROPIC_API_KEY`.) The frontier model usually finishes in 2 steps; the local 7B model often takes 4-5 and sometimes loops on adjacent files. **You're feeling the model-quality axis directly** — same code, same tools, different brain.

## 6. The trace IS the agent

In `--show-trace` output, look at each step's `reasoning` line. Ask yourself:

- Did the model articulate a sub-goal, or did it just narrate?
- Did the next action serve the reasoning, or contradict it?
- Did the model's reasoning evolve as tool outputs came back, or did it stay rigid?

> A coding agent is only as good as its reasoning trace. If you can't read the trace and recognize a competent engineer's thinking, your prompts need work.

## 7. Stretch — programmatic API

```python
from lantern import agent_loop

result = agent_loop(
    "How does the agent loop terminate when the model can't decide?",
    repo=".",
    max_steps=4,
)
print(result.answer)
print(f"\nSteps used: {len(result.steps)}  forced_final={result.forced_final}")
for i, s in enumerate(result.steps, 1):
    print(f"  {i}. {s.decision.next_action}: {s.decision.reasoning}")
```

`AgentResult` is the right shape for an eval harness. In week 7, we'll grade these traces directly: "did the agent answer correctly, AND did it cite the right files?"

---

When you're done, you've built the actual core of every coding agent in 2026. Week 7 makes it production-grade — caching, traces, MCP server.

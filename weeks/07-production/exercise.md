# Week 7 Exercise (5 min)

Install Lantern into Claude Code as an MCP server, then watch it work.

## 1. Verify the MCP server boots

```bash
# Direct stdio handshake (just to prove the server starts)
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}' \
  | lantern mcp 2>/dev/null | head -1
```

You should see a JSON line with `"protocolVersion"` in it. That's the server saying "ready to serve tools."

## 2. Save a trace

Run any multi-step ask with `--save-trace`:

```bash
lantern ask "Where is reciprocal rank fusion implemented?" --save-trace
```

Note the `trace saved: <run_id>` line. Then:

```bash
lantern trace                  # see it in the list
lantern trace <run_id>         # pretty replay
```

Open the raw file too:

```bash
ls -la ~/.lantern/traces/
cat ~/.lantern/traces/<run_id>.jsonl | jq .
```

Each line is one event. **This is what every production agent ships.**

## 3. Wire it into Claude Code

Open `~/Library/Application Support/Claude/claude_desktop_config.json` and add:

```json
{
  "mcpServers": {
    "lantern": {
      "command": "lantern",
      "args": ["mcp"],
      "env": {
        "LANTERN_REPO": "/Users/you/Desktop/projects/ai-engineering-stack"
      }
    }
  }
}
```

(If the file doesn't exist, create it with that single object. If `mcpServers` already has entries, add `"lantern"` alongside them.)

Restart Claude Code. In a new conversation, ask:

> "What does the lantern package expose? Use the lantern_search tool to find out."

Claude Code should pick `lantern_search`, get back ranked hits, then probably read a file and answer. **Lantern is now part of Claude Code's tool surface.**

## 4. Drive Claude Code through every Lantern tool

Try these prompts in Claude Code:

- "Use lantern_about_repo to tell me which project I'm working in."
- "Run lantern_summarize_file on lantern/agent.py — what does it do?"
- "Use lantern_ask to answer: where is BM25 search implemented?"
- "List the public API of lantern with lantern_list_dir and lantern_search."

Each call goes through your Lantern stdio server, runs locally on your Mac, returns to Claude Code. **Zero cloud round-trip for the actual work.**

## 5. Compare traces from a successful and a failing question

```bash
# Likely succeeds
lantern ask "Where is hybrid_search defined?" --save-trace --show-trace

# Likely degenerates (model loops on broad grep)
lantern ask "Trace the entire flow from user prompt to final answer." --save-trace --show-trace --max-steps 4
```

Compare the two trace replays:

```bash
lantern trace
# pick the two run_ids
lantern trace <success_id>
lantern trace <fail_id>
```

What's different? Usually:
- The success has 1-2 steps, distinct reasoning, exits via `answer`.
- The failure has duplicate `next_action` calls and exits via `forced_final`.

> **The trace tells you exactly what to fix.** This is the rarest skill in AI engineering — debugging an agent's reasoning by reading its work.

## 6. Stretch — point Lantern at a different repo

```bash
lantern index ~/some/other/repo
LANTERN_REPO=~/some/other/repo lantern mcp
```

Then update the Claude Code config with the new path, restart, and ask Claude Code questions about *that* repo. **One Lantern install can serve any number of projects** — just spin up another stdio server with a different `LANTERN_REPO`.

---

When you've wired Claude Code to call your local Lantern server and inspected at least one full trace, you've shipped a real AI infrastructure piece. Week 8 turns Lantern into a published artifact: public benchmark across Qwen 7B vs Claude vs GPT, blog post, demo video.
